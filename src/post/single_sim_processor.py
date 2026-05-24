#!/usr/bin/env python3
# single_sim_processor.py

import csv
import glob
import os
import pickle
import subprocess
import sys
import time

import numpy as np

from src.post.metrics import MetricComputer, MetricFormatter
from src.post.output_manager import OutputManager
from src.post.simulation_plotter import SimulationPlotter
from src.post.thermal_adapter import ThermalAdapter, ThermalAdapterError
from src.post.visualize_mapping import ChipletVisualizer


class SingleSimProcessor:
    """
    Processes results from a single simulation run.
    
    This class handles loading simulation results, computing metrics,
    formatting outputs, and generating plots and visualizations.
    """
    
    # Default empty system results for comparison
    DEFAULT_EMPTY_SYSTEM_RESULTS = {
    }
    
    def __init__(self, raw_results_dir, processing_config):
        """
        Initialize the single simulation processor.
        
        Args:
            raw_results_dir (str): Path to the raw results directory
            processing_config (dict): Processing configuration containing:
                - warmup_period_us (float): Warmup period in microseconds
                - run_wkld_agg_comm (bool): Run workload aggregate communication simulation
                - run_ind_comm (bool): Run individual layer communication simulation
                - run_net_agg_comm (bool): Run network aggregate communication simulation
                - generate_plots (bool): Generate plots
                - generate_visualizations (bool): Generate visualizations
        """
        self.raw_results_dir = raw_results_dir
        self.processing_config = processing_config
        
        # Extract processing parameters
        self.warmup_period = processing_config.get('warmup_period_us', 0.0)
        self.cooldown_period = processing_config.get('cooldown_period_us', 0.0)
        self.run_wkld_agg_comm = processing_config.get('run_wkld_agg_comm', False)
        self.run_ind_comm = processing_config.get('run_ind_comm', False)
        self.run_net_agg_comm = processing_config.get('run_net_agg_comm', False)
        self.generate_plots = processing_config.get('generate_plots', True)
        self.generate_visualizations = processing_config.get('generate_visualizations', False)
        
        # Initialize state
        self.gm = None
        self.formatted_results_dir = None
        self.metric_computer = None
        self.thermal_adapter_result = None
        
    def process(self):
        """
        Execute the complete post-processing workflow.
        
        Returns:
            dict: Processing results containing metric_computer and results directory
        """
        print("\n" + "="*80)
        print("📊 STARTING POST-PROCESSING")
        print("="*80)
        print(f"📁 Raw Results Directory: {self.raw_results_dir}")
        
        start_time = time.time()
        
        # Load simulation state
        self._load_simulation_state()
        
        # Apply warmup filtering
        self._apply_warmup_filter()
        
        # Create formatted results directory
        self._create_formatted_results_dir()
        
        # Run supplementary simulations
        self._run_supplementary_simulations()
        
        # Compute metrics
        self._compute_metrics()
        
        # Format and save metrics
        self._format_and_save_metrics()
        
        # Save power data
        self._save_power_data()

        # Run thermal integration once formatted output and power metrics exist
        self._run_thermal_if_enabled()
        
        # Generate plots
        if self.generate_plots:
            self._generate_plots()
        
        # Generate visualizations
        if self.generate_visualizations:
            self._generate_visualizations()
        
        # Print summary
        duration = time.time() - start_time
        self._print_summary(duration)
        
        return {
            'metric_computer': self.metric_computer,
            'results_dir': self.formatted_results_dir,
            'gm': self.gm
        }
    
    def _load_simulation_state(self):
        """Load the simulation state from the pickle file."""
        print("\n🔄 Loading simulation state...")
        
        # Find the .pkl file that contains the serialized GlobalManager state.
        # Ignore auxiliary pickle files (e.g., dsent_stats.pkl) that now live alongside it.
        pkl_files = [
            path for path in glob.glob(os.path.join(self.raw_results_dir, "*.pkl"))
            if "dsent_stats" not in os.path.basename(path)
        ]
        
        if len(pkl_files) != 1:
            raise RuntimeError(
                f"Expected exactly 1 .pkl file in {self.raw_results_dir}, "
                f"but found {len(pkl_files)}: {pkl_files}"
            )
        
        pkl_file = pkl_files[0]
        print(f"📦 Loading from: {pkl_file}")
        
        load_start = time.time()
        try:
            with open(pkl_file, 'rb') as f:
                self.gm = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load simulation state from {pkl_file}: {e}")
        
        load_duration = time.time() - load_start
        print(f"✅ State loaded successfully ({load_duration:.2f}s)")
        
        # Migrate legacy attributes if needed
        if hasattr(self.gm, '_migrate_legacy_attributes'):
            self.gm._migrate_legacy_attributes()
    
    def _apply_warmup_filter(self):
        """Apply warmup period filtering to retired models."""
        print(f"\nℹ️  Applying warmup period: {self.warmup_period} μs")
        
        self.gm.warmup_period_us = self.warmup_period
        # Strictly require Model-based method
        if not hasattr(self.gm, 'filter_retired_models_by_warmup'):
            raise AttributeError("GlobalManager missing required method 'filter_retired_models_by_warmup'")
        self.gm.filter_retired_models_by_warmup()
        
        # Optionally apply cooldown filtering after warmup
        if getattr(self, 'cooldown_period', 0.0) and self.cooldown_period > 0:
            print(f"ℹ️  Applying cooldown period: {self.cooldown_period} μs")
            self.gm.post_warmup_retired_models = self.gm.temporal_filter.filter_by_cooldown(
                retired_models=self.gm.post_warmup_retired_models,
                cooldown_period_us=self.cooldown_period,
                simulation_end_time_us=self.gm.global_time_us
            )
        
        num_models = len(self.gm.post_warmup_retired_models)
        print(f"✅ Warmup filter applied: {num_models} models remaining")
    
    def _create_formatted_results_dir(self):
        """Create the directory for formatted results."""
        print("\n📁 Creating formatted results directory...")
        
        # Get base formatted results directory
        base_formatted_dir = os.path.join(os.getcwd(), "_results", "formatted_results")
        
        # Use the raw results directory name as the formatted directory name
        raw_dir_name = os.path.basename(self.raw_results_dir)
        self.formatted_results_dir = os.path.join(base_formatted_dir, raw_dir_name)
        
        os.makedirs(self.formatted_results_dir, exist_ok=True)
        print(f"✅ Formatted results directory: {self.formatted_results_dir}")
    
    def _run_supplementary_simulations(self):
        """Run supplementary communication simulations if configured."""
        self.workload_aggregate_results = None
        self.individual_results = None
        self.network_aggregate_results = None
        
        if self.run_wkld_agg_comm:
            print("\n🔄 Running workload aggregate communication simulation...")
            start = time.time()
            self.workload_aggregate_results = self.gm.comm_simulator.simulate_workload_aggregate_communication(
                self.gm.retired_mapped_models
            )
            duration = time.time() - start
            print(f"✅ Workload aggregate simulation completed ({duration:.2f}s)")
        
        if self.run_ind_comm:
            print("\n🔄 Running individual layer communication simulation...")
            start = time.time()
            self.individual_results = self.gm.comm_simulator.simulate_individual_layer_communication(
                self.gm.retired_mapped_models
            )
            duration = time.time() - start
            print(f"✅ Individual layer simulation completed ({duration:.2f}s)")
        
        if self.run_net_agg_comm:
            print("\n🔄 Running network aggregate communication simulation...")
            start = time.time()
            self.network_aggregate_results = self.gm.comm_simulator.simulate_network_aggregate_communication(
                self.gm.retired_mapped_models
            )
            duration = time.time() - start
            print(f"✅ Network aggregate simulation completed ({duration:.2f}s)")
    
    def _compute_metrics(self):
        """Compute all metrics from the simulation results."""
        print("\n⚙️  Computing metrics...")
        start = time.time()
        
        # Locate the DSENT stats file. The SimulationRunner now stores the final
        # path on the GlobalManager before serialization, but we keep a small
        # fallback for legacy/raw runs.
        dsent_stats_path = getattr(self.gm, 'dsent_stats_file', None)
        if not dsent_stats_path or not os.path.exists(dsent_stats_path):
            pkl_path = os.path.join(self.raw_results_dir, 'dsent_stats.pkl')
            json_path = os.path.join(self.raw_results_dir, 'dsent_stats.jsonl')
            if os.path.exists(pkl_path):
                dsent_stats_path = pkl_path
            elif os.path.exists(json_path):
                dsent_stats_path = json_path
            else:
                dsent_stats_path = pkl_path  # default path for warning messages
        
        # Initialize metric computer
        self.metric_computer = MetricComputer(
            self.gm.post_warmup_retired_models,
            self.gm.global_time_us,
            self.gm.system.num_chiplets,
            dsent_stats_file_path=dsent_stats_path
        )
        
        # Compute utilization metrics
        self.metric_computer.compute_avg_system_utilization()
        self.metric_computer.compute_utilization_over_time(self.gm.time_step_us)
        
        # Compute model summary metrics
        model_summary_metrics = self.metric_computer.compute_model_summary_metrics()
        print(f"   📈 Generated summary metrics for {len(model_summary_metrics)} models")
        
        # Compute approach comparison metrics
        self.metric_computer.compute_approach_comparison_metrics(
            self.individual_results,
            self.DEFAULT_EMPTY_SYSTEM_RESULTS
        )
        
        # Compute power and energy profiles
        print("   🔋 Computing power profile...")
        self.metric_computer.compute_power_profile(time_step_us=self.gm.time_step_us)
        
        print("   ⚡ Computing energy profile...")
        self.metric_computer.compute_energy_metrics()
        
        duration = time.time() - start
        print(f"✅ Metrics computed successfully ({duration:.2f}s)")
    
    def _format_and_save_metrics(self):
        """Format metrics and save to files."""
        print("\n💾 Formatting and saving metrics...")
        start = time.time()
        
        # Create output manager
        output_manager = OutputManager(
            wl_file_name=os.path.basename(self.gm.workload_manager.wl_file),
            adj_matrix_file=os.path.basename(self.gm.adj_matrix_file),
            chiplet_mapping_file=os.path.basename(self.gm.chiplet_mapping_file),
            communication_simulator=self.gm.communication_simulator,
            communication_method=self.gm.communication_method,
            mapping_function=self.gm.mapping_function,
            metric_computer=self.metric_computer,
            results_dir=self.formatted_results_dir,
            num_chiplets=self.gm.system.num_chiplets
        )
        
        # Create metric formatter
        metric_formatter = MetricFormatter(
            metric_computer=self.metric_computer,
            global_manager=self.gm
        )
        
        # Format all metrics
        formatted_model_metrics = metric_formatter.format_all_model_metrics()
        formatted_utilization_metrics = metric_formatter.format_utilization_metrics(self.gm.time_step_us)
        formatted_comparison_metrics = metric_formatter.format_approach_comparison_metrics()
        
        wall_clock_runtime = self.gm.wall_clock_runtime_s if hasattr(self.gm, 'wall_clock_runtime_s') else 0
        formatted_simulation_summary = metric_formatter.format_simulation_summary(wall_clock_runtime)
        formatted_energy_metrics = metric_formatter.format_energy_metrics()
        
        # Save all formatted metrics
        output_manager.save_formatted_metrics(formatted_model_metrics, subdirectory="formatted_model_metrics")
        output_manager.save_formatted_metrics(formatted_utilization_metrics, subdirectory="formatted_utilization_metrics")
        output_manager.save_formatted_metrics(formatted_comparison_metrics, subdirectory="formatted_comparison_metrics")
        output_manager.save_formatted_metrics(formatted_simulation_summary, subdirectory=None)
        output_manager.save_formatted_metrics(formatted_energy_metrics, subdirectory="formatted_energy_metrics")
        
        duration = time.time() - start
        print(f"✅ Metrics saved successfully ({duration:.2f}s)")
    
    def _save_power_data(self):
        """Save chiplet power data averaged over 100-microsecond intervals."""
        power_data = self.metric_computer.chiplet_total_power_over_time
        
        if not power_data:
            print("\nℹ️  No power data to save")
            return
        
        print("\n💾 Saving 100-microsecond averaged chiplet power data...")
        
        power_csv_path = os.path.join(self.formatted_results_dir, 'chiplet_power_100us_avg.csv')
        time_step_us = self.gm.time_step_us
        
        if time_step_us <= 0:
            print("⚠️  Invalid time step, cannot average power data")
            return
        
        steps_per_100us = int(100 / time_step_us)
        
        if steps_per_100us <= 0:
            print("⚠️  Time step too large to average over 100us intervals")
            return
        
        sorted_chiplet_ids = sorted(power_data.keys())
        max_power = 140.0
        
        with open(power_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for chiplet_id in sorted_chiplet_ids:
                chiplet_name = f"chiplet_chiplet_{chiplet_id}_chiplet"
                power_values = power_data[chiplet_id]
                
                # Pad array to be a multiple of steps_per_100us
                num_values = len(power_values)
                num_to_pad = (steps_per_100us - num_values % steps_per_100us) % steps_per_100us
                
                if num_to_pad > 0:
                    padded_values = np.pad(power_values, (0, num_to_pad), mode='constant', constant_values=np.nan)
                else:
                    padded_values = power_values
                
                # Reshape and compute mean
                reshaped_values = padded_values.reshape(-1, steps_per_100us)
                averaged_power = np.nanmean(reshaped_values, axis=1)
                power_percentages = (averaged_power / max_power) * 100
                
                writer.writerow([chiplet_name] + list(power_percentages))
        
        print(f"✅ Power data saved to: {power_csv_path}")

    def _run_thermal_if_enabled(self):
        """Prepare thermal-model inputs when post-processing thermal output is enabled."""
        thermal_config = self.processing_config.get('thermal', {})
        if not isinstance(thermal_config, dict) or not thermal_config.get('enabled', False):
            return

        print("\n🌡️  Preparing thermal model inputs...")
        try:
            adapter = ThermalAdapter.from_metric_computer(
                formatted_results_dir=self.formatted_results_dir,
                time_step_us=self.gm.time_step_us,
                metric_computer=self.metric_computer,
                thermal_config=thermal_config,
            )
            self.thermal_adapter_result = adapter.prepare_inputs()
        except ThermalAdapterError as exc:
            self._write_thermal_summary(
                status='FAILED',
                thermal_config=thermal_config,
                wall_clock_runtime_s=0.0,
                error_message=str(exc),
            )
            if self._thermal_fail_on_error(thermal_config):
                raise
            print(f"⚠️  Thermal input generation failed: {exc}")
            return
        except Exception as exc:
            self._write_thermal_summary(
                status='FAILED',
                thermal_config=thermal_config,
                wall_clock_runtime_s=0.0,
                error_message=f"Thermal input generation failed: {exc}",
            )
            if self._thermal_fail_on_error(thermal_config):
                raise
            print(f"⚠️  Thermal input generation failed: {exc}")
            return

        print(f"✅ Thermal inputs prepared in: {self.thermal_adapter_result.thermal_output_dir}")
        self._invoke_thermal_solver(thermal_config)

    def _invoke_thermal_solver(self, thermal_config):
        """Invoke thermal_RC.py with generated CHIPSIM thermal inputs."""
        thermal_model_dir = os.path.join(os.getcwd(), 'integrations', 'thermal_model')
        log_path = self._thermal_log_file()
        start_time = time.time()

        print("🌡️  Running thermal solver...")
        try:
            command = self._build_thermal_command(thermal_config)
            os.makedirs(self._thermal_output_dir(), exist_ok=True)
            with open(log_path, 'w') as log_file:
                log_file.write('CHIPSIM thermal solver invocation\n')
                log_file.write(f"Working directory: {thermal_model_dir}\n")
                log_file.write(f"Command: {' '.join(command)}\n\n")
                log_file.flush()
                subprocess.run(
                    command,
                    cwd=thermal_model_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        except subprocess.CalledProcessError as exc:
            runtime_s = time.time() - start_time
            error_message = f"Thermal solver failed with exit code {exc.returncode}."
            self._write_thermal_summary(
                status='FAILED',
                thermal_config=thermal_config,
                wall_clock_runtime_s=runtime_s,
                error_message=error_message,
                log_path=log_path,
            )
            if self._thermal_fail_on_error(thermal_config):
                raise
            print(f"⚠️  {error_message} See: {log_path}")
            return
        except OSError as exc:
            runtime_s = time.time() - start_time
            error_message = f"Failed to start thermal solver: {exc}"
            self._write_thermal_summary(
                status='FAILED',
                thermal_config=thermal_config,
                wall_clock_runtime_s=runtime_s,
                error_message=error_message,
                log_path=log_path,
            )
            if self._thermal_fail_on_error(thermal_config):
                raise
            print(f"⚠️  {error_message}")
            return
        except Exception as exc:
            runtime_s = time.time() - start_time
            error_message = f"Thermal execution failed: {exc}"
            self._write_thermal_summary(
                status='FAILED',
                thermal_config=thermal_config,
                wall_clock_runtime_s=runtime_s,
                error_message=error_message,
                log_path=log_path,
            )
            if self._thermal_fail_on_error(thermal_config):
                raise
            print(f"⚠️  {error_message}")
            return

        runtime_s = time.time() - start_time
        self._write_thermal_summary(
            status='SUCCESS',
            thermal_config=thermal_config,
            wall_clock_runtime_s=runtime_s,
            log_path=log_path,
        )
        print("✅ Thermal solver completed")

    def _build_thermal_command(self, thermal_config):
        """Build the thermal_RC.py subprocess command without shell interpolation."""
        thermal_model_dir = os.path.join(os.getcwd(), 'integrations', 'thermal_model')
        thermal_script = os.path.join(thermal_model_dir, 'thermal_RC.py')

        time_step_s = self._thermal_time_step_s(thermal_config)
        power_interval_s = self._thermal_power_interval_s(thermal_config, time_step_s)
        total_duration_s = self._thermal_total_duration_s(thermal_config, power_interval_s)
        time_heatmap_s = self._thermal_time_heatmap_s(thermal_config, total_duration_s)

        return [
            sys.executable,
            thermal_script,
            '--material_prop_file', str(thermal_config['material_prop_file']),
            '--geometry_file', str(thermal_config['geometry_file']),
            '--power_config_file', str(self.thermal_adapter_result.resolved_power_config_file),
            '--power_seq_file', str(self.thermal_adapter_result.power_sequence_file),
            '--output_dir', str(self.thermal_adapter_result.thermal_output_dir),
            '--simulation_type', str(thermal_config.get('simulation_type', 'transient')),
            '--time_step', str(time_step_s),
            '--power_interval', str(power_interval_s),
            '--total_duration', str(total_duration_s),
            '--time_heatmap', str(time_heatmap_s),
            '--generate_heatmap', self._thermal_bool_arg(thermal_config.get('generate_heatmap', True)),
            '--generate_DSS', self._thermal_bool_arg(thermal_config.get('generate_DSS', False)),
            '--use_tuned_C', self._thermal_bool_arg(thermal_config.get('use_tuned_C', True)),
        ]

    def _write_thermal_summary(
        self,
        status,
        thermal_config,
        wall_clock_runtime_s,
        error_message=None,
        log_path=None,
    ):
        """Write a compact thermal run summary for CHIPSIM artifacts."""
        thermal_output_dir = self._thermal_output_dir()
        os.makedirs(thermal_output_dir, exist_ok=True)
        summary_path = self._thermal_summary_file()
        kpis = self._collect_thermal_kpis()
        log_path = log_path or self._thermal_log_file()

        with open(summary_path, 'w') as f:
            f.write('CHIPSIM Thermal Summary\n')
            f.write('=======================\n')
            f.write(f"Status: {status}\n")
            f.write(f"Thermal output directory: {thermal_output_dir}\n")
            f.write(f"Thermal log: {log_path}\n")
            f.write(f"Wall-clock runtime (s): {wall_clock_runtime_s:.6f}\n")
            f.write(f"Simulation type: {thermal_config.get('simulation_type', 'transient')}\n")
            f.write(f"Generated heatmaps: {thermal_config.get('generate_heatmap', True)}\n")
            f.write(f"Generated DSS matrices: {thermal_config.get('generate_DSS', False)}\n")
            f.write(f"Fail on error: {self._thermal_fail_on_error(thermal_config)}\n")

            if error_message:
                f.write(f"Error: {error_message}\n")
                error_snippet = self._read_file_tail(log_path)
                if error_snippet:
                    f.write('\nError log tail:\n')
                    f.write(error_snippet)
                    if not error_snippet.endswith('\n'):
                        f.write('\n')

            f.write('\nThermal KPIs\n')
            f.write('------------\n')
            self._write_temperature_summary_line(f, 'Global peak temperature', kpis.get('global_peak_temperature_k'))
            self._write_temperature_summary_line(f, 'Final timestep peak temperature', kpis.get('final_timestep_peak_temperature_k'))
            self._write_temperature_summary_line(f, 'Per-timestep global peak min', kpis.get('timestep_global_peak_min_k'))
            self._write_temperature_summary_line(f, 'Per-timestep global peak avg', kpis.get('timestep_global_peak_avg_k'))
            self._write_temperature_summary_line(f, 'Per-timestep global peak max', kpis.get('timestep_global_peak_max_k'))

            f.write('\nDerived max_power_w by chiplet\n')
            f.write('------------------------------\n')
            for chiplet_id, max_power_w in self._sorted_max_power_items():
                f.write(f"chiplet_{chiplet_id}: {float(max_power_w):.12g} W\n")

            f.write('\nZero-power chiplets\n')
            f.write('-------------------\n')
            zero_power_chiplets = self._zero_power_chiplets()
            if zero_power_chiplets:
                for chiplet_id in zero_power_chiplets:
                    f.write(f"chiplet_{chiplet_id}\n")
            else:
                f.write('None\n')

            f.write('\nFile pointers\n')
            f.write('-------------\n')
            self._write_optional_file_pointer(f, 'Power sequence', self._adapter_attr('power_sequence_file'))
            self._write_optional_file_pointer(f, 'Resolved power config', self._adapter_attr('resolved_power_config_file'))
            self._write_optional_file_pointer(f, 'Adapter metadata', self._adapter_attr('adapter_metadata_file'))
            self._write_optional_file_pointer(f, 'Floorplan directory', os.path.join(thermal_output_dir, 'floorplan'))
            self._write_optional_file_pointer(f, 'Heatmaps directory', os.path.join(thermal_output_dir, 'heatmaps'))
            self._write_optional_file_pointer(f, 'Thermal output directory', os.path.join(thermal_output_dir, 'output'))
            self._write_file_list(f, 'Temperature files', kpis.get('temperature_files', []))
            self._write_file_list(f, 'Heatmap files', kpis.get('heatmap_files', []))
            self._write_file_list(f, 'DSS matrix files', kpis.get('dss_matrix_files', []))

    def _collect_thermal_kpis(self):
        """Collect temperature KPIs and thermal-native output file pointers."""
        thermal_output_dir = self._thermal_output_dir()
        output_dir = os.path.join(thermal_output_dir, 'output')
        temperature_files = sorted(glob.glob(os.path.join(output_dir, 'temperature_all_*.csv')))
        heatmap_files = sorted(glob.glob(os.path.join(thermal_output_dir, 'heatmaps', '*.png')))
        dss_matrix_files = [
            path for path in [
                os.path.join(output_dir, 'disc_A_matrix.csv'),
                os.path.join(output_dir, 'disc_B_matrix.csv'),
            ]
            if os.path.exists(path)
        ]

        global_peak = None
        final_timestep_peak = None
        timestep_peaks = []

        for temperature_file in temperature_files:
            try:
                temperature_data = np.loadtxt(temperature_file, delimiter=',')
            except (OSError, ValueError):
                continue

            if temperature_data.size == 0:
                continue

            temperature_data = np.atleast_2d(temperature_data)
            finite_temperatures = temperature_data[np.isfinite(temperature_data)]
            if finite_temperatures.size == 0:
                continue

            file_peak = float(np.max(finite_temperatures))
            global_peak = file_peak if global_peak is None else max(global_peak, file_peak)

            file_timestep_peaks = np.nanmax(temperature_data, axis=1)
            file_timestep_peaks = file_timestep_peaks[np.isfinite(file_timestep_peaks)]
            if file_timestep_peaks.size > 0:
                timestep_peaks.extend(float(value) for value in file_timestep_peaks)
                final_timestep_peak = float(file_timestep_peaks[-1])

        kpis = {
            'temperature_files': temperature_files,
            'heatmap_files': heatmap_files,
            'dss_matrix_files': dss_matrix_files,
            'global_peak_temperature_k': global_peak,
            'final_timestep_peak_temperature_k': final_timestep_peak,
        }

        if timestep_peaks:
            kpis.update({
                'timestep_global_peak_min_k': float(np.min(timestep_peaks)),
                'timestep_global_peak_avg_k': float(np.mean(timestep_peaks)),
                'timestep_global_peak_max_k': float(np.max(timestep_peaks)),
            })

        return kpis

    def _thermal_time_step_s(self, thermal_config):
        value = thermal_config.get('time_step_s')
        if value is not None:
            return float(value)
        return float(self.gm.time_step_us) * 1e-6

    def _thermal_power_interval_s(self, thermal_config, time_step_s):
        value = thermal_config.get('power_interval_s')
        if value is not None:
            return float(value)
        return time_step_s

    def _thermal_total_duration_s(self, thermal_config, power_interval_s):
        value = thermal_config.get('total_duration_s')
        if value is not None:
            return float(value)
        return self._num_power_time_points() * power_interval_s

    def _thermal_time_heatmap_s(self, thermal_config, total_duration_s):
        value = thermal_config.get('time_heatmap_s')
        if value is not None:
            return float(value)
        return total_duration_s

    def _num_power_time_points(self):
        power_traces = self.metric_computer.chiplet_total_power_over_time or {}
        for trace in power_traces.values():
            return len(trace)
        return 0

    def _thermal_output_dir(self):
        if self.thermal_adapter_result is not None:
            return self.thermal_adapter_result.thermal_output_dir
        return os.path.join(self.formatted_results_dir, 'thermal')

    def _thermal_log_file(self):
        return os.path.join(self._thermal_output_dir(), 'thermal.log')

    def _thermal_summary_file(self):
        return os.path.join(self._thermal_output_dir(), 'thermal_summary.txt')

    def _adapter_attr(self, attr_name):
        if self.thermal_adapter_result is None:
            return None
        return getattr(self.thermal_adapter_result, attr_name, None)

    def _sorted_max_power_items(self):
        if self.thermal_adapter_result is None:
            return []
        items = self.thermal_adapter_result.max_power_w_by_chiplet.items()
        return sorted(items, key=lambda item: self._chiplet_sort_key(item[0]))

    def _zero_power_chiplets(self):
        if self.thermal_adapter_result is None:
            return []
        return sorted(self.thermal_adapter_result.zero_power_chiplets, key=self._chiplet_sort_key)

    @staticmethod
    def _chiplet_sort_key(chiplet_id):
        try:
            return (0, int(chiplet_id))
        except (TypeError, ValueError):
            return (1, str(chiplet_id))

    def _thermal_fail_on_error(self, thermal_config):
        return self._thermal_bool_value(thermal_config.get('fail_on_error', True))

    @staticmethod
    def _thermal_bool_value(value):
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        return bool(value)

    @staticmethod
    def _thermal_bool_arg(value):
        return 'true' if SingleSimProcessor._thermal_bool_value(value) else 'false'

    @staticmethod
    def _write_temperature_summary_line(summary_file, label, value_k):
        if value_k is None:
            summary_file.write(f"{label}: unavailable\n")
            return
        summary_file.write(f"{label}: {value_k:.6f} K ({value_k - 273.15:.6f} C)\n")

    @staticmethod
    def _write_optional_file_pointer(summary_file, label, path):
        if path:
            summary_file.write(f"{label}: {path}\n")

    @staticmethod
    def _write_file_list(summary_file, label, paths):
        summary_file.write(f"{label}:\n")
        if paths:
            for path in paths:
                summary_file.write(f"  {path}\n")
        else:
            summary_file.write('  None\n')

    @staticmethod
    def _read_file_tail(path, max_chars=2000):
        if not path or not os.path.exists(path):
            return ''
        with open(path, 'r', errors='replace') as f:
            content = f.read()
        return content[-max_chars:]

    def _generate_plots(self):
        """Generate plots from the simulation results."""
        print("\n📊 Generating plots...")
        start = time.time()
        
        plotter = SimulationPlotter(
            results_folder=self.formatted_results_dir,
            metric_computer=self.metric_computer
        )
        
        plotter.plot_utilization_over_time()
        plotter.plot_approach_comparison_metrics()
        plotter.plot_power_over_time()
        
        duration = time.time() - start
        print(f"✅ Plots generated successfully ({duration:.2f}s)")
    
    def _generate_visualizations(self):
        """Generate visualizations of network mappings and system state."""
        print("\n🎨 Generating visualizations...")
        start = time.time()
        
        visualizer = ChipletVisualizer(
            adj_matrix_file=self.gm.adj_matrix_file,
            results_folder=self.formatted_results_dir
        )
        
        network_viz_dir = os.path.join(self.formatted_results_dir, "network_mapping_visualizations")
        os.makedirs(network_viz_dir, exist_ok=True)
        
        visualizer.visualize_network_mappings_from_data(
            retired_mapped_models=self.gm.post_warmup_retired_models,
            output_dir=network_viz_dir
        )
        
        visualizer.visualize_system_state_over_time(
            retired_mapped_models=self.gm.post_warmup_retired_models,
            output_dir=network_viz_dir
        )
        
        duration = time.time() - start
        print(f"✅ Visualizations generated successfully ({duration:.2f}s)")
    
    def _print_summary(self, duration):
        """Print processing summary."""
        print("\n" + "="*80)
        print("🏁 POST-PROCESSING SUMMARY")
        print("="*80)
        print(f"⏱️  Processing Time:        {duration:.2f} seconds")
        print(f"📁 Formatted Results:      {self.formatted_results_dir}")
        print(f"📊 Models Processed:       {len(self.gm.post_warmup_retired_models)}")
        print(f"🔋 Plots Generated:        {self.generate_plots}")
        print(f"🎨 Visualizations Created: {self.generate_visualizations}")
        print("="*80)
