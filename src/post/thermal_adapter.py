#!/usr/bin/env python3
# thermal_adapter.py
"""
Thermal adapter for converting CHIPSIM post-processing state into thermal-model inputs.

This module owns the boundary between CHIPSIM-native power data and the thermal
model input contract.
"""

import csv
import json
import os
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple


POWER_SEQUENCE_FILENAME = 'power_seq_from_chipsim.csv'
RESOLVED_POWER_CONFIG_FILENAME = 'power_config_resolved.yml'
ADAPTER_METADATA_FILENAME = 'thermal_adapter_metadata.json'
V1_GRANULARITY = 'chiplet_total'
V1_BLOCK_NAME = 'total'


class ThermalAdapterError(RuntimeError):
    """Raised when CHIPSIM data cannot be adapted for thermal modeling."""


@dataclass(frozen=True)
class ThermalAdapterInputs:
    """Inputs required to prepare thermal-model files from a CHIPSIM run."""

    formatted_results_dir: str
    time_step_us: float
    chiplet_power_over_time: Mapping[int, Sequence[float]]
    thermal_config: Mapping[str, Any]


@dataclass(frozen=True)
class ThermalAdapterResult:
    """Paths and metadata produced by the thermal adapter."""

    thermal_output_dir: str
    power_sequence_file: str
    resolved_power_config_file: Optional[str]
    adapter_metadata_file: str
    max_power_w_by_chiplet: Mapping[int, float]
    zero_power_chiplets: Sequence[int]


@dataclass(frozen=True)
class PowerTraceConversion:
    """Thermal percent rows derived from CHIPSIM watt traces."""

    rows: Sequence[Sequence[Any]]
    max_power_w_by_chiplet: Mapping[int, float]
    zero_power_chiplets: Sequence[int]
    chiplet_metadata: Mapping[str, Mapping[str, Any]]
    num_time_points: int


def extract_chiplet_power_over_time(metric_computer: Any) -> Mapping[int, Sequence[float]]:
    """Read chiplet total power traces from a MetricComputer after power metrics run."""
    power_traces = getattr(metric_computer, 'chiplet_total_power_over_time', None)

    if power_traces is None:
        raise ThermalAdapterError(
            "MetricComputer.chiplet_total_power_over_time is not available. "
            "Call MetricComputer.compute_power_profile() before preparing thermal inputs."
        )

    if not isinstance(power_traces, MappingABC):
        raise ThermalAdapterError(
            "MetricComputer.chiplet_total_power_over_time must be a mapping of "
            "chiplet_id to power trace."
        )

    if not power_traces:
        raise ThermalAdapterError(
            "MetricComputer.chiplet_total_power_over_time is empty; no chiplet power "
            "traces are available for thermal input generation."
        )

    for chiplet_id, power_trace in power_traces.items():
        if power_trace is None:
            raise ThermalAdapterError(
                f"Power trace for chiplet {chiplet_id!r} is missing."
            )

    return dict(power_traces)


def chiplet_name(chiplet_id: Any) -> str:
    """Return the v1 thermal chiplet name for a CHIPSIM chiplet ID."""
    return f"chiplet_{chiplet_id}"


def power_row_key(power_layer: str, chiplet_id: Any, block_name: str = V1_BLOCK_NAME) -> str:
    """Return the thermal power-sequence row key for a v1 chiplet-total block."""
    return f"{power_layer}_{chiplet_name(chiplet_id)}_{block_name}"


def convert_watts_to_percent_rows(
    chiplet_power_over_time: Mapping[int, Sequence[float]],
    power_layer: str,
    block_name: str = V1_BLOCK_NAME
) -> PowerTraceConversion:
    """Convert per-chiplet watt traces to thermal percent-of-max rows."""
    rows = []
    max_power_w_by_chiplet = {}
    zero_power_chiplets = []
    chiplet_metadata = {}
    expected_num_time_points = None

    for chiplet_id in sorted(chiplet_power_over_time, key=_chiplet_sort_key):
        power_trace_w = _coerce_power_trace(chiplet_id, chiplet_power_over_time[chiplet_id])

        if expected_num_time_points is None:
            expected_num_time_points = len(power_trace_w)
        elif len(power_trace_w) != expected_num_time_points:
            raise ThermalAdapterError(
                "All chiplet power traces must have the same length. "
                f"Expected {expected_num_time_points} samples, but chiplet {chiplet_id!r} "
                f"has {len(power_trace_w)} samples."
            )

        max_power_w = max(power_trace_w)
        max_power_w_by_chiplet[chiplet_id] = max_power_w
        row_key = power_row_key(power_layer, chiplet_id, block_name)

        if max_power_w == 0:
            percent_values = [0.0 for _ in power_trace_w]
            zero_power_chiplets.append(chiplet_id)
        else:
            percent_values = [(power_w / max_power_w) * 100.0 for power_w in power_trace_w]

        rows.append([row_key] + [_format_percent(value) for value in percent_values])
        chiplet_metadata[str(chiplet_id)] = {
            'chiplet_id': str(chiplet_id),
            'chiplet_name': chiplet_name(chiplet_id),
            'block_name': block_name,
            'row_key': row_key,
            'max_power_w': max_power_w,
            'num_samples': len(power_trace_w),
            'zero_power': max_power_w == 0,
        }

    return PowerTraceConversion(
        rows=rows,
        max_power_w_by_chiplet=max_power_w_by_chiplet,
        zero_power_chiplets=zero_power_chiplets,
        chiplet_metadata=chiplet_metadata,
        num_time_points=expected_num_time_points or 0,
    )


def _chiplet_sort_key(chiplet_id: Any) -> Tuple[int, Any]:
    try:
        return (0, int(chiplet_id))
    except (TypeError, ValueError):
        return (1, str(chiplet_id))


def _coerce_power_trace(chiplet_id: Any, power_trace: Sequence[float]) -> List[float]:
    try:
        values = [float(value) for value in power_trace]
    except TypeError as exc:
        raise ThermalAdapterError(
            f"Power trace for chiplet {chiplet_id!r} must be an iterable of watt values."
        ) from exc
    except ValueError as exc:
        raise ThermalAdapterError(
            f"Power trace for chiplet {chiplet_id!r} contains a non-numeric value."
        ) from exc

    if not values:
        raise ThermalAdapterError(f"Power trace for chiplet {chiplet_id!r} is empty.")

    negative_values = [value for value in values if value < 0]
    if negative_values:
        raise ThermalAdapterError(
            f"Power trace for chiplet {chiplet_id!r} contains negative watt values."
        )

    return values


def _format_percent(value: float) -> str:
    return f"{value:.12g}"


def _load_yaml(path: str) -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ThermalAdapterError(
            "PyYAML is required to resolve thermal power_config_file. "
            "Install the CHIPSIM thermal/config dependencies before generating thermal inputs."
        ) from exc

    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _write_yaml(path: str, data: Any) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ThermalAdapterError(
            "PyYAML is required to write thermal power_config_resolved.yml. "
            "Install the CHIPSIM thermal/config dependencies before generating thermal inputs."
        ) from exc

    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _clone_without_aliases(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {key: _clone_without_aliases(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_without_aliases(item) for item in value]
    if isinstance(value, tuple):
        return [_clone_without_aliases(item) for item in value]
    return value


class ThermalAdapter:
    """Prepares thermal-model inputs from CHIPSIM post-processing data."""

    def __init__(self, inputs: ThermalAdapterInputs):
        self.inputs = inputs

    @classmethod
    def from_metric_computer(
        cls,
        formatted_results_dir: str,
        time_step_us: float,
        metric_computer: Any,
        thermal_config: Mapping[str, Any]
    ):
        """Build an adapter by reading power traces from a computed MetricComputer."""
        return cls(
            ThermalAdapterInputs(
                formatted_results_dir=formatted_results_dir,
                time_step_us=time_step_us,
                chiplet_power_over_time=extract_chiplet_power_over_time(metric_computer),
                thermal_config=thermal_config,
            )
        )

    @property
    def thermal_output_dir(self) -> str:
        """Return the per-run thermal output directory path."""
        return os.path.join(self.inputs.formatted_results_dir, 'thermal')

    @property
    def power_sequence_file(self) -> str:
        """Return the generated thermal power sequence path."""
        return os.path.join(self.thermal_output_dir, POWER_SEQUENCE_FILENAME)

    @property
    def resolved_power_config_file(self) -> str:
        """Return the generated thermal power config path."""
        return os.path.join(self.thermal_output_dir, RESOLVED_POWER_CONFIG_FILENAME)

    @property
    def adapter_metadata_file(self) -> str:
        """Return the generated adapter metadata path."""
        return os.path.join(self.thermal_output_dir, ADAPTER_METADATA_FILENAME)

    def prepare_inputs(self) -> ThermalAdapterResult:
        """Prepare thermal-model inputs from CHIPSIM power traces."""
        os.makedirs(self.thermal_output_dir, exist_ok=True)
        power_layer = self._required_config_value('power_layer')
        granularity = self.inputs.thermal_config.get('granularity', V1_GRANULARITY)

        if granularity != V1_GRANULARITY:
            raise ThermalAdapterError(
                f"Unsupported thermal granularity {granularity!r}; v1 supports {V1_GRANULARITY!r}."
            )

        conversion = convert_watts_to_percent_rows(
            self.inputs.chiplet_power_over_time,
            power_layer=power_layer,
            block_name=V1_BLOCK_NAME,
        )
        self._write_power_sequence(conversion.rows)
        self._write_resolved_power_config(conversion.max_power_w_by_chiplet, power_layer)
        self._write_adapter_metadata(conversion, power_layer, granularity)

        return ThermalAdapterResult(
            thermal_output_dir=self.thermal_output_dir,
            power_sequence_file=self.power_sequence_file,
            resolved_power_config_file=self.resolved_power_config_file,
            adapter_metadata_file=self.adapter_metadata_file,
            max_power_w_by_chiplet=conversion.max_power_w_by_chiplet,
            zero_power_chiplets=conversion.zero_power_chiplets,
        )

    def _required_config_value(self, key: str) -> Any:
        value = self.inputs.thermal_config.get(key)
        if value in (None, ''):
            raise ThermalAdapterError(f"Missing required thermal config key: {key}")
        return value

    def _write_power_sequence(self, rows: Sequence[Sequence[Any]]) -> None:
        with open(self.power_sequence_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def _write_resolved_power_config(
        self,
        max_power_w_by_chiplet: Mapping[int, float],
        power_layer: str
    ) -> None:
        power_config_file = self._required_config_value('power_config_file')
        power_config = _clone_without_aliases(_load_yaml(power_config_file))

        if not isinstance(power_config, MappingABC):
            raise ThermalAdapterError(
                f"Thermal power_config_file must contain a YAML mapping: {power_config_file}"
            )

        layer_config = power_config.get(power_layer)
        if not isinstance(layer_config, MappingABC):
            raise ThermalAdapterError(
                f"Thermal power_config_file does not contain layer {power_layer!r}: "
                f"{power_config_file}"
            )

        for chiplet_id, max_power_w in max_power_w_by_chiplet.items():
            chiplet_config = layer_config.get(chiplet_name(chiplet_id))
            if not isinstance(chiplet_config, MappingABC):
                raise ThermalAdapterError(
                    f"Thermal power_config_file is missing chiplet {chiplet_name(chiplet_id)!r} "
                    f"under layer {power_layer!r}."
                )

            layout_blocks = chiplet_config.get('layout_blocks')
            if not isinstance(layout_blocks, MappingABC):
                raise ThermalAdapterError(
                    f"Thermal power_config_file chiplet {chiplet_name(chiplet_id)!r} must define "
                    f"layout_blocks for v1 granularity {V1_GRANULARITY!r}."
                )

            block_config = layout_blocks.get(V1_BLOCK_NAME)
            if not isinstance(block_config, MappingABC):
                raise ThermalAdapterError(
                    f"Thermal power_config_file chiplet {chiplet_name(chiplet_id)!r} must define "
                    f"block {V1_BLOCK_NAME!r} for v1 granularity {V1_GRANULARITY!r}."
                )

            block_config['max_power'] = float(max_power_w)

        _write_yaml(self.resolved_power_config_file, power_config)

    def _write_adapter_metadata(
        self,
        conversion: PowerTraceConversion,
        power_layer: str,
        granularity: str
    ) -> None:
        metadata = {
            'thermal_output_dir': self.thermal_output_dir,
            'power_sequence_file': self.power_sequence_file,
            'resolved_power_config_file': self.resolved_power_config_file,
            'power_layer': power_layer,
            'granularity': granularity,
            'block_name': V1_BLOCK_NAME,
            'time_step_us': self.inputs.time_step_us,
            'num_time_points': conversion.num_time_points,
            'max_power_w_by_chiplet': {
                str(chiplet_id): max_power_w
                for chiplet_id, max_power_w in conversion.max_power_w_by_chiplet.items()
            },
            'zero_power_chiplets': [str(chiplet_id) for chiplet_id in conversion.zero_power_chiplets],
            'chiplets': conversion.chiplet_metadata,
        }

        with open(self.adapter_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write('\n')
