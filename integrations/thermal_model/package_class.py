import common 
from layer_class import Layer_chiplet
import numpy as np
import os
from scipy.signal import lti
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import expm_multiply
from ctypes import *
import platform

# load the shared C library, for the entire c-based solver
# Here check the shared_library extension and load the library accordingly
os_type = platform.system()
if os_type == 'Windows':
    shared_lib_ext = '.dll'
elif os_type == 'Darwin':  # for macOS dylib works
    shared_lib_ext = '.dylib'
else:  # Assume Unix/Linux
    shared_lib_ext = '.so'

solver_dir = os.path.dirname(os.path.abspath(__file__))
lib_solver = CDLL(os.path.join(solver_dir, 'c_files', f'chiplet_ode{shared_lib_ext}'))

lib_solver.chiplet_ode.argtypes = [
    POINTER(POINTER(c_double)),  # output temperature
    POINTER(POINTER(c_double)),  # power input
    POINTER(POINTER(c_double)),  # G_all
    POINTER(POINTER(c_int)),   # non_zero_indexes
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),   # C
    c_int,  # total_nodes
    c_int,  # non_zero_columns
    c_double, # total_duration
    c_double, # time_step
    c_double, # power_interval
]

lib_solver.chiplet_ode.restype = None

def chiplet_ode_c(output_temperature, power_input, G_all, non_zero_indexes, C, total_nodes, non_zero_columns, total_duration, time_step, power_interval):

    lib_solver.chiplet_ode(output_temperature, power_input, G_all, non_zero_indexes, C, total_nodes, non_zero_columns, total_duration, time_step, power_interval)

class Chiplet_package:
    def __init__(self, material_properties, geometry_dict, power_grid_class, args, utils=common.Utils):
        self.material_properties = material_properties
        self.geometry_dict = geometry_dict
        self.layers = []
        self.common_utils = utils
        self.power_grid_class = power_grid_class
        self.args = args
    
    def create_layers(self):
        for layer in self.geometry_dict['layers']:
            self.layers.append(Layer_chiplet(name=layer, 
                                             layer_dict=self.geometry_dict['layers'][layer], 
                                             material_properties=self.material_properties,
                                             power_grid_class=self.power_grid_class,
                                             args=self.args))
            
        for layer in self.layers:
            layer.create_nodes(utils=self.common_utils, 
                               material_properties=self.material_properties)
    
    def connect_nodes(self):
        # create the thermal network, create a 1D numpy array of nodes with thermal capacitance.
        # create a 2D array of thermal resistance between the nodes
        
        # 1. each layer would create 1D and 2D array for capacitance and resistance. 
            # layer would return 1D array of capacitanace, 2D array of resistance, 
            # 1D array of Z resistance, 1D array of area for Z direction and x,y coordinates of the nodes
        # 2. connect the layers for Z direction, using the Z resistance and area overlap
        # 3. Add a ground node to connect top and bottom layer with the convection resistance

        # Three resistances: layer's internal, layer's with top and bottom layer and convection resistance

        # connects nodes and calculate the RC between the nodes, and convection resistance
        for layer in self.layers:
            layer.connect_nodes()

        self.shared_conductance_list = []
        # connect the layers for Z direction
        for i in range(len(self.layers)-1):
            bottom_layer = self.layers[i]
            top_layer = self.layers[i+1]
            shared_conductance = self.connect_layers(top_layer, bottom_layer)
            self.shared_conductance_list.append(shared_conductance)

        self.capacitance_all = []
        for layer in self.layers:
            self.capacitance_all.append(layer.get_capacitance())
        
        self.capacitance_all = np.concatenate(self.capacitance_all, axis=0)
        
        # invert the capacitance
        self.capacitance_all = 1/self.capacitance_all

        self.conductance_all = np.zeros((self.package_total_nodes(), self.package_total_nodes() + 1))
        layer_global_iter = 0
        for layer_iter in range(len(self.layers)):
            current_layer = self.layers[layer_iter]
            convective_cond = current_layer.get_convective_conductance()
            xy_conductance = current_layer.get_conductance()
            current_layer_length = current_layer.layer_total_nodes()

            if layer_iter == 0: 
                bottom_layer = 0
                bottom_layer_length = 0
                bottom_conductance = np.zeros(1)

                top_layer = self.layers[layer_iter+1]
                top_layer_length = top_layer.layer_total_nodes()
                top_conductance = self.shared_conductance_list[layer_iter]
            
            elif layer_iter == len(self.layers)-1:
                top_layer = 0
                top_layer_length = 0
                top_conductance = np.zeros(1)

                bottom_layer = self.layers[layer_iter-1]
                bottom_layer_length = bottom_layer.layer_total_nodes()
                bottom_conductance = np.transpose(self.shared_conductance_list[layer_iter-1])

            else:
                top_layer = self.layers[layer_iter+1]
                top_layer_length = top_layer.layer_total_nodes()
                top_conductance = self.shared_conductance_list[layer_iter]

                bottom_layer = self.layers[layer_iter-1]
                bottom_layer_length = bottom_layer.layer_total_nodes()
                bottom_conductance = np.transpose(self.shared_conductance_list[layer_iter-1])
            
            for layer_node_iter in range(current_layer_length):
                node_iter = layer_global_iter + layer_node_iter
                
                # convective resistance
                self.conductance_all[node_iter, -1] = convective_cond[layer_node_iter]
                
                # layer resistance
                self.conductance_all[node_iter, layer_global_iter:layer_global_iter+current_layer_length] = xy_conductance[layer_node_iter]

                # bottom layer resistance
                if layer_iter != 0:
                    self.conductance_all[node_iter, layer_global_iter - bottom_layer_length:layer_global_iter] = bottom_conductance[layer_node_iter]
                
                # top layer resistance
                if layer_iter != len(self.layers)-1:
                    self.conductance_all[node_iter, layer_global_iter + current_layer_length:layer_global_iter + current_layer_length + top_layer_length] = top_conductance[layer_node_iter]

            layer_global_iter += current_layer_length

        conductance_diag = np.sum(self.conductance_all, axis=1)
        conductance_diag = np.diag(conductance_diag)

        self.conductance_all = self.conductance_all[:,:-1]
        self.conductance_all = conductance_diag - self.conductance_all
        
        if self.args.use_tuned_C:
            self.apply_tuned_C()
        else:
            pass

        if self.args.generate_DSS:
            self.generate_DSS()

        non_zero_conductance = [row[row != 0] for row in self.conductance_all]
        non_zero_index = [list(np.nonzero(row)[0]) for row in self.conductance_all]
        max_len = max(len(row) for row in non_zero_conductance)

        self.non_zero_index = np.array([row + [row[-1]] * (max_len - len(row)) for row in non_zero_index])
        self.conductance_all = np.zeros((self.conductance_all.shape[0], max_len))

        for i, row in enumerate(non_zero_conductance):
            self.conductance_all[i, :len(row)] = row

    def connect_layers(self, top_layer, bottom_layer):
        shared_conductance = np.zeros((bottom_layer.layer_total_nodes(), top_layer.layer_total_nodes()))
        
        for i in range(bottom_layer.layer_total_nodes()):
            for j in range(top_layer.layer_total_nodes()):
                # calculate overlapping area
                common_area = common.calculate_overlapping_area(x1=bottom_layer.x_cordinates[i], y1=bottom_layer.y_cordinates[i], 
                                                                x2=top_layer.x_cordinates[j], y2=top_layer.y_cordinates[j], 
                                                                x_len1=bottom_layer.x_lengths[i], y_len1=bottom_layer.y_lengths[i], 
                                                                x_len2=top_layer.x_lengths[j], y_len2=top_layer.y_lengths[j])
                
                if common_area > 0:
                    shared_conductance[i,j] = (common_area*bottom_layer.z_conductance[i]*top_layer.z_conductance[j])/(bottom_layer.z_conductance[i]*top_layer.xy_area[j] + top_layer.z_conductance[j]*bottom_layer.xy_area[i])
                    # shared_conductance[i,j] = (top_layer.z_resistance[i]*top_layer.xy_area[i] + bottom_layer.z_resistance[j]*bottom_layer.xy_area[j])/(common_area)

        return shared_conductance

    def apply_tuned_C(self):
        intial_C_guess = np.array([1.35551358, 1.3345646, 0.46572207, 0.85322922, 2.0361129, 1.77131198, 2.0619255, 0.94317305, 0.6672266])

        num_nodes = 0

        for layer in self.layers:
            if 'substrate_1' in layer.layer_name:
                layer_start = num_nodes
                num_nodes += layer.layer_total_nodes()
                layer_end = num_nodes
                self.capacitance_all[layer_start:layer_end] = intial_C_guess[0]*self.capacitance_all[layer_start:layer_end]
            
            elif 'substrate_2' in layer.layer_name:
                layer_start = num_nodes
                num_nodes += layer.layer_total_nodes()
                layer_end = num_nodes
                self.capacitance_all[layer_start:layer_end] = intial_C_guess[1]*self.capacitance_all[layer_start:layer_end] 
            
            elif 'c4' in layer.layer_name:
                layer_start = num_nodes
                num_nodes += layer.layer_total_nodes()
                layer_end = num_nodes
                self.capacitance_all[layer_start:layer_end] = intial_C_guess[2]*self.capacitance_all[layer_start:layer_end]

            elif 'interposer' in layer.layer_name:
                layer_start = num_nodes
                num_nodes += layer.layer_total_nodes()
                layer_end = num_nodes
                self.capacitance_all[layer_start:layer_end] = intial_C_guess[3]*self.capacitance_all[layer_start:layer_end]
            
            elif 'ubump' in layer.layer_name:
                layer_start = num_nodes
                num_nodes += layer.layer_total_nodes()
                layer_end = num_nodes
                self.capacitance_all[layer_start:layer_end] = intial_C_guess[4]*self.capacitance_all[layer_start:layer_end]
            
            elif 'chiplet' in layer.layer_name:
                layer_start = num_nodes
                num_nodes += layer.layer_total_nodes()
                layer_end = num_nodes
                self.capacitance_all[layer_start:layer_end] = intial_C_guess[5]*self.capacitance_all[layer_start:layer_end]
            
            elif 'tim' in layer.layer_name:
                layer_start = num_nodes
                num_nodes += layer.layer_total_nodes()
                layer_end = num_nodes
                self.capacitance_all[layer_start:layer_end] = intial_C_guess[6]*self.capacitance_all[layer_start:layer_end]
            
            elif 'lid1' in layer.layer_name:
                layer_start = num_nodes
                num_nodes += layer.layer_total_nodes()
                layer_end = num_nodes
                self.capacitance_all[layer_start:layer_end] = intial_C_guess[7]*self.capacitance_all[layer_start:layer_end]
            
            elif 'lid2' in layer.layer_name:
                layer_start = num_nodes
                num_nodes += layer.layer_total_nodes()
                layer_end = num_nodes
                self.capacitance_all[layer_start:layer_end] = intial_C_guess[8]*self.capacitance_all[layer_start:layer_end]
    
    def generate_floorplan(self):
        # generate the floorplan of the package, and chiplet
        num_nodes = 0
        for layer in self.layers:
            layer.plot_layer(utils=self.common_utils, layer_start=num_nodes)
            num_nodes += layer.layer_total_nodes()

    def package_total_nodes(self):
        total_nodes = 0
        for layer in self.layers:
            total_nodes += layer.layer_total_nodes()
        return total_nodes
    
    def generate_DSS(self):
        # generate the A and B matrix for DSS
        # A = -C^-1*G
        # B = C^-1
        # C = I
        # D = 0
        capacitance_matrix = np.diag(self.capacitance_all)
        A = -(capacitance_matrix @ self.conductance_all)
        B = capacitance_matrix
        C = np.eye(self.package_total_nodes())
        D = np.zeros((self.package_total_nodes(), self.package_total_nodes()))

        l_sys = lti(A, B, C, D)
        d_sys = l_sys.to_discrete(self.args.time_step, method='zoh')

        discrete_A = d_sys.A
        discrete_B = d_sys.B
        
        if not os.path.exists(self.args.output_dir + '/output'):
            os.makedirs(self.args.output_dir + '/output')

        np.savetxt(self.args.output_dir + '/output/disc_A_matrix.csv', discrete_A, delimiter=',')
        np.savetxt(self.args.output_dir + '/output/disc_B_matrix.csv', discrete_B, delimiter=',')

    def write_temperature_to_file(self, ts):
        self.temperature_all_save = np.array(self.temperature_all_save) + 300.0

        # save the temperature to a file
        file_name = f'{self.args.output_dir}/output/temperature_all_{ts}.csv'
        np.savetxt(file_name, self.temperature_all_save, delimiter=',')

        
        temperature_all_map = self.temperature_all_save.T

        if self.args.generate_heatmap:
            index_heatmap = self._heatmap_step_index(temperature_all_map.shape[1])
            plot_temperature = temperature_all_map[:, index_heatmap] - 300.0
            num_nodes = 0
            for layer in self.layers:
                layer_start = num_nodes
                num_nodes += layer.layer_total_nodes()
                layer_end = num_nodes
                layer.plot_heatmap(plot_temperature[layer_start:layer_end], utils=self.common_utils)

        num_nodes = 0
        for layer in self.layers:
            layer_start = num_nodes
            num_nodes += layer.layer_total_nodes()
            layer_end = num_nodes
            if layer.is_power_src():
                layer.map_temperature_to_blk(temperature_all_map[layer_start:layer_end], utils=self.common_utils, ts=ts)

    def convert_to_np_array(self, pointer, rows=None):
        def dereference_pointer(pointer, length):
            double_array = cast(pointer, POINTER(c_double * length))
            double_list = list(double_array.contents)
            return double_list

        if rows is None:
            rows = len(pointer)
        self.temperature_all_save = [
            dereference_pointer(pointer[i], self.package_total_nodes())
            for i in range(rows)
        ]

    def set_initial_conditions(self):
        if not os.path.exists(self.args.output_dir + '/output'):
            os.makedirs(self.args.output_dir + '/output')

        if self.args.simulation_type == 'steady':
            power_steps = 1
        else:
            power_steps = self._count_steps(self.args.total_duration, self.args.power_interval, 'total_duration/power_interval')
        self.temperature_all_save = []
        self.temperature_all = np.zeros(self.package_total_nodes())
        self.power = np.zeros((self.package_total_nodes(), power_steps))

        # set power for chiplet nodes
        global_iter = 0
        for layer in self.layers:
            self.power[global_iter:global_iter+layer.layer_total_nodes()] = layer.get_power(power_steps)
            global_iter += layer.layer_total_nodes()

        # export power to csv file
        np.savetxt(self.args.output_dir + '/output/power_all.csv', self.power.T, delimiter=',')

    def run_simulation_c_lsoda(self):
        total_duration = self.args.total_duration
        self.set_initial_conditions()
        power = self.power.T

        if self.args.simulation_type == 'steady':
            dt = float(total_duration)
            power_interval = float(total_duration)
            self._validate_native_inputs(power, dt, power_interval)
            self.temperature_all_save = self._run_steady_constant_power(power, total_duration)
        else:
            dt = float(self.args.time_step)
            power_interval = float(self.args.power_interval)
            self._validate_native_inputs(power, dt, power_interval)
            max_native_steps = self._max_native_steps()

            if max_native_steps > 0 and power.shape[0] > max_native_steps:
                self.temperature_all_save = self._run_transient_chunked(
                    power=power,
                    dt=dt,
                    power_interval=power_interval,
                    max_native_steps=max_native_steps,
                )
            else:
                output_rows = self._count_steps(total_duration, dt, 'total_duration/time_step') + 1
                initial_temperature = np.zeros(self.package_total_nodes())
                self.temperature_all_save = self._call_native_solver(
                    power=power,
                    output_rows=output_rows,
                    initial_temperature=initial_temperature,
                    total_duration=total_duration,
                    dt=dt,
                    power_interval=power_interval,
                )

        self.write_temperature_to_file(dt)

    def _run_steady_constant_power(self, power, total_duration):
        total_nodes = self.package_total_nodes()
        initial_temperature = np.zeros(total_nodes)
        power_vector = np.asarray(power[-1], dtype=np.double)

        conductance = self._sparse_conductance_matrix()
        capacitance_inverse = np.asarray(self.capacitance_all, dtype=np.double)
        system_matrix = -(diags(capacitance_inverse, format='csr') @ conductance)
        forcing = capacitance_inverse * power_vector

        augmented_matrix = vstack([
            hstack([system_matrix, csr_matrix(forcing.reshape(-1, 1))], format='csr'),
            csr_matrix((1, total_nodes + 1)),
        ], format='csr')
        augmented_initial = np.concatenate([initial_temperature, [1.0]])
        final_temperature = expm_multiply(augmented_matrix * total_duration, augmented_initial)[:-1]

        return np.vstack([initial_temperature, final_temperature])

    def _sparse_conductance_matrix(self):
        total_nodes = self.package_total_nodes()
        non_zero_columns = self.conductance_all.shape[1]
        row_index = np.repeat(np.arange(total_nodes), non_zero_columns)
        column_index = self.non_zero_index.reshape(-1)
        values = self.conductance_all.reshape(-1)
        populated = values != 0
        return csr_matrix(
            (values[populated], (row_index[populated], column_index[populated])),
            shape=(total_nodes, total_nodes),
        )

    def _run_transient_chunked(self, power, dt, power_interval, max_native_steps):
        if not np.isclose(dt, power_interval, rtol=1e-9, atol=1e-12):
            raise ValueError(
                'Chunked thermal solving requires time_step and power_interval to match; '
                f'got time_step={dt} and power_interval={power_interval}. '
                'Set --max_native_steps 0 to use the unchunked native solver.'
            )

        total_steps = power.shape[0]
        total_chunks = int(np.ceil(total_steps / float(max_native_steps)))
        print(f'Running native thermal solver in {total_chunks} chunks of up to {max_native_steps} steps')

        stitched_segments = []
        initial_temperature = np.zeros(self.package_total_nodes())

        for chunk_index, start in enumerate(range(0, total_steps, max_native_steps), start=1):
            end = min(start + max_native_steps, total_steps)
            segment_power = power[start:end]
            segment_steps = end - start
            print(f'Thermal native chunk {chunk_index}/{total_chunks}: steps {start + 1}-{end}')

            segment_temperature = self._call_native_solver(
                power=segment_power,
                output_rows=segment_steps + 1,
                initial_temperature=initial_temperature,
                total_duration=segment_steps * dt,
                dt=dt,
                power_interval=power_interval,
            )
            segment_temperature += initial_temperature - segment_temperature[0]

            if stitched_segments:
                stitched_segments.append(segment_temperature[1:])
            else:
                stitched_segments.append(segment_temperature)
            initial_temperature = segment_temperature[-1].copy()

        return np.vstack(stitched_segments)

    def _call_native_solver(self, power, output_rows, initial_temperature, total_duration, dt, power_interval):
        total_nodes = self.package_total_nodes()
        np_temperature_all = np.zeros((output_rows, total_nodes))
        np_temperature_all[0] = np.asarray(initial_temperature, dtype=np.double)

        c_temperature_all = (POINTER(c_double) * output_rows)()
        temperature_rows = []
        for i in range(output_rows):
            row = (c_double * total_nodes)(*np_temperature_all[i])
            temperature_rows.append(row)
            c_temperature_all[i] = row

        power = np.ascontiguousarray(power, dtype=np.double)
        if power.shape[0] < output_rows:
            pad_rows = output_rows - power.shape[0]
            power = np.vstack([power, np.repeat(power[-1:], pad_rows, axis=0)])
        c_power = (POINTER(c_double) * power.shape[0])()
        power_rows = []
        for i in range(power.shape[0]):
            row = (c_double * power.shape[1])(*power[i])
            power_rows.append(row)
            c_power[i] = row

        G_all = np.ascontiguousarray(self.conductance_all, dtype=np.double)
        non_zero_index = np.ascontiguousarray(self.non_zero_index, dtype=np.intc)

        c_G_all = (POINTER(c_double) * G_all.shape[0])()
        conductance_rows = []
        for i in range(G_all.shape[0]):
            row = (c_double * G_all.shape[1])(*G_all[i])
            conductance_rows.append(row)
            c_G_all[i] = row

        c_non_zero_index = (POINTER(c_int) * non_zero_index.shape[0])()
        non_zero_index_rows = []
        for i in range(non_zero_index.shape[0]):
            row = (c_int * non_zero_index.shape[1])(*non_zero_index[i])
            non_zero_index_rows.append(row)
            c_non_zero_index[i] = row

        capacitance = np.ascontiguousarray(self.capacitance_all, dtype=np.double)

        chiplet_ode_c(
            c_temperature_all,
            c_power,
            c_G_all,
            c_non_zero_index,
            capacitance,
            total_nodes,
            non_zero_index.shape[1],
            total_duration,
            dt,
            power_interval,
        )

        self.convert_to_np_array(c_temperature_all, rows=output_rows)
        return np.asarray(self.temperature_all_save, dtype=np.double)

    def _heatmap_step_index(self, output_columns):
        if self.args.simulation_type == 'steady':
            return output_columns - 1

        index = self._count_steps(
            self.args.time_heatmap,
            self.args.time_step,
            'time_heatmap/time_step',
        )
        if index < 0 or index >= output_columns:
            raise ValueError(
                f'time_heatmap index {index} is outside available temperature steps '
                f'0..{output_columns - 1}'
            )
        return index

    def _validate_native_inputs(self, power, dt, power_interval):
        total_nodes = self.package_total_nodes()
        if dt <= 0:
            raise ValueError(f'time_step must be positive, got {dt}')
        if power_interval <= 0:
            raise ValueError(f'power_interval must be positive, got {power_interval}')
        if power.shape[1] != total_nodes:
            raise ValueError(f'Power matrix has {power.shape[1]} nodes, expected {total_nodes}')
        if self.args.simulation_type != 'steady':
            expected_power_steps = self._count_steps(
                self.args.total_duration,
                self.args.power_interval,
                'total_duration/power_interval',
            )
            if power.shape[0] != expected_power_steps:
                raise ValueError(f'Power matrix has {power.shape[0]} rows, expected {expected_power_steps}')
            self._count_steps(self.args.total_duration, self.args.time_step, 'total_duration/time_step')

        self._require_finite_array('capacitance_all', self.capacitance_all)
        self._require_finite_array('conductance_all', self.conductance_all)
        self._require_finite_array('non_zero_index', self.non_zero_index)
        self._require_finite_array('power', power)

        if np.any(self.capacitance_all == 0):
            raise ValueError('capacitance_all contains zero values')
        if np.any(self.non_zero_index < 0) or np.any(self.non_zero_index >= total_nodes):
            raise ValueError('non_zero_index contains node indexes outside package bounds')

    def _max_native_steps(self):
        value = getattr(self.args, 'max_native_steps', 256)
        if value is None:
            return 256
        value = int(value)
        if value < 0:
            raise ValueError(f'max_native_steps must be non-negative, got {value}')
        return value

    @staticmethod
    def _require_finite_array(name, values):
        if not np.all(np.isfinite(values)):
            raise ValueError(f'{name} contains NaN or infinite values')

    @staticmethod
    def _count_steps(duration, interval, label):
        raw_steps = float(duration) / float(interval)
        steps = int(round(raw_steps))
        if not np.isclose(raw_steps, steps, rtol=1e-9, atol=1e-9):
            raise ValueError(f'{label} must be an integer number of steps, got {raw_steps}')
        return steps
