import csv
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import common

class Power_block:
    def __init__(self, power_node_dict, name, chiplet_x, chiplet_y, args):
        self.name = name
        self.x = power_node_dict['start_point_x'] + chiplet_x
        self.y = power_node_dict['start_point_y'] + chiplet_y
        self.length_x = power_node_dict['length_x']
        self.length_y = power_node_dict['length_y']
        self.max_power = float(power_node_dict['max_power'])
        self.args = args

    def create_power_seq(self, layer_name, chiplet_name, power_seq):
        found_flag = False
        power_sq_name = layer_name + '_' + chiplet_name + '_' + self.name
        for i in range(0, len(power_seq)):
            if power_seq[i][0] == power_sq_name:
                self.power_seq = power_seq[i][1:]
                found_flag = True
                break
        
        if not found_flag:
            raise ValueError(f'Power sequence not found for block: {power_sq_name}')
        
        # power seq is in percentage, convert to float
        try:
            power_seq_percent = [float(i) for i in self.power_seq]
        except ValueError as exc:
            raise ValueError(f'Power sequence for block {power_sq_name} contains a non-numeric value') from exc

        if self.args.simulation_type == 'steady':
            step_index = self._steady_power_step_index(len(power_seq_percent))
            self.power_seq = np.array([power_seq_percent[step_index]/100.0])
        else:
            self.power_seq = np.array([i/100.0 for i in power_seq_percent])

        # print
        # print('Power sequence for node: ', layer , '_', self.name, ' is: ', self.power_seq)

    def get_area(self):
        return self.length_x * self.length_y

    def _steady_power_step_index(self, sequence_length):
        if sequence_length <= 0:
            return 0

        try:
            time_heatmap = float(self.args.time_heatmap)
            time_step = float(self.args.time_step)
        except (AttributeError, TypeError, ValueError):
            return sequence_length - 1

        if time_step <= 0:
            return sequence_length - 1

        step_index = int(round(time_heatmap/time_step))
        return min(max(step_index, 0), sequence_length - 1)

class Power_chiplet:
    def __init__(self, power_chiplet_dict, name, args):
        self.name = name
        self.power_blocks = []
        self.args = args
        self.chiplet_x = power_chiplet_dict['start_chiplet_x']
        self.chiplet_y = power_chiplet_dict['start_chiplet_y']
        self.is_power_src = True
        if not self.args.is_homogeneous:
            self.length_chiplet_x = power_chiplet_dict['length_chiplet_x']
            self.length_chiplet_y = power_chiplet_dict['length_chiplet_y']
            self.nodes_x = power_chiplet_dict['nodes_x']
            self.nodes_y = power_chiplet_dict['nodes_y']

            if 'material' in power_chiplet_dict:
                self.material = power_chiplet_dict['material']
            
        if 'layout_blocks' in power_chiplet_dict:
            for block in power_chiplet_dict['layout_blocks']:
                power_block = Power_block(power_chiplet_dict['layout_blocks'][block], block, self.chiplet_x, self.chiplet_y, self.args)
                self.power_blocks.append(power_block)
        else:
            self.is_power_src = False
    
    def create_power_seq_chiplet(self, layer_name, power_seq):
        for block in self.power_blocks:
            block.create_power_seq(layer_name, self.name, power_seq)

class Power_layer:
    def __init__(self, power_layer_dict, layer, args):
        self.name = layer
        self.power_chiplets = []
        self.args = args
        for chiplet in power_layer_dict[layer]:
            power_chiplet = Power_chiplet(power_layer_dict[layer][chiplet], chiplet, self.args)
            self.power_chiplets.append(power_chiplet)
        
        # sort the chiplets based on x and y
        sorted_by_y = sorted(self.power_chiplets, key=lambda chiplet: chiplet.chiplet_y)
        self.power_chiplets = sorted(sorted_by_y, key=lambda chiplet: chiplet.chiplet_x)

        # for chiplet in self.power_chiplets:
        #     print(chiplet.chiplet_x, chiplet.chiplet_y)

    def create_power_seq_layer(self, power_seq):
        for chiplet in self.power_chiplets:
            chiplet.create_power_seq_chiplet(self.name, power_seq)
    
    def plot_layer(self, utils):
        fig, ax = plt.subplots(figsize=(6,6))
        
        for power_chiplet in self.power_chiplets:
            for power_block in power_chiplet.power_blocks:
                rect = Rectangle((power_block.x , power_block.y), 
                                    power_block.length_x, power_block.length_y, 
                                    fc="none", ec="black", lw=0.5)
                ax.add_patch(rect)
                name = power_chiplet.name + '-' + power_block.name
                # name = power_block.name
                plt.plot(power_block.x, power_block.y, 'ro', markersize=0.5)
                plt.text(power_block.x, power_block.y, name, fontsize=5)

        ax.set_xlim(-0.5, utils.package_x_len + 0.5)
        ax.set_ylim(-0.5, utils.package_y_len + 0.5)

        plt.title('Power grid for ' + self.name)
        plt.xlabel('X dimension (mm)')
        plt.ylabel('Y dimension (mm)')

        # check if floorplan directory exists, if not create it

        if not os.path.exists(self.args.output_dir + '/floorplan'):
            os.makedirs(self.args.output_dir + '/floorplan')

        fig.savefig(self.args.output_dir + '/floorplan/' + self.name + '_power_' + '.png', dpi=300, bbox_inches='tight')
        plt.close()


class Power_grid:
    def __init__(self, power_dict, args):
        self.power_dict = power_dict
        self.args = args

        with open(self.args.power_seq_file, 'r') as f:
            reader = csv.reader(f)
            self.power_seq = [row for row in reader if row]
        
        self.power_layers = []
        for layer in self.power_dict:
            power_layer = Power_layer(self.power_dict, layer, self.args)
            self.power_layers.append(power_layer)

        self._validate_power_sequence()
    
    def create_power_seq_grid(self, utils):
        for layer in self.power_layers:
            layer.plot_layer(utils)
            layer.create_power_seq_layer(self.power_seq)

    def _validate_power_sequence(self):
        if not self.power_seq:
            raise ValueError(f'Power sequence file is empty: {self.args.power_seq_file}')

        row_by_name = {}
        sequence_steps = None

        for row_number, row in enumerate(self.power_seq, start=1):
            row[0] = row[0].strip()
            row_name = row[0]
            if not row_name:
                raise ValueError(f'Power sequence row {row_number} has an empty block name')
            if row_name in row_by_name:
                raise ValueError(
                    f'Duplicate power sequence row for block {row_name}: '
                    f'rows {row_by_name[row_name]} and {row_number}'
                )
            row_by_name[row_name] = row_number

            row_steps = len(row) - 1
            if row_steps <= 0:
                raise ValueError(f'Power sequence row {row_name} has no timestep values')
            if sequence_steps is None:
                sequence_steps = row_steps
            elif row_steps != sequence_steps:
                raise ValueError(
                    f'Power sequence row {row_name} has {row_steps} steps, '
                    f'expected {sequence_steps}'
                )

            for value_index, value in enumerate(row[1:], start=1):
                try:
                    numeric_value = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f'Power sequence row {row_name}, step {value_index} '
                        f'is not numeric: {value!r}'
                    ) from exc
                if not np.isfinite(numeric_value):
                    raise ValueError(
                        f'Power sequence row {row_name}, step {value_index} '
                        f'is not finite: {value!r}'
                    )

        expected_names = self._expected_power_sequence_names()
        expected_name_set = set(expected_names)
        missing_names = [name for name in expected_names if name not in row_by_name]
        extra_names = sorted(name for name in row_by_name if name not in expected_name_set)

        if missing_names:
            raise ValueError(
                f'Power sequence is missing {len(missing_names)} required rows: '
                f'{self._preview_names(missing_names)}'
            )
        if extra_names:
            raise ValueError(
                f'Power sequence has {len(extra_names)} rows not present in the power config: '
                f'{self._preview_names(extra_names)}'
            )

        if self.args.simulation_type == 'steady':
            self._validate_steady_heatmap_index(sequence_steps)
        else:
            expected_steps = self._count_steps(
                self.args.total_duration,
                self.args.power_interval,
                'total_duration/power_interval',
            )
            if sequence_steps != expected_steps:
                raise ValueError(
                    f'Power sequence has {sequence_steps} steps, expected {expected_steps} '
                    f'from total_duration/power_interval '
                    f'({self.args.total_duration}/{self.args.power_interval})'
                )

    def _expected_power_sequence_names(self):
        names = []
        for layer in self.power_layers:
            for chiplet in layer.power_chiplets:
                for block in chiplet.power_blocks:
                    names.append(layer.name + '_' + chiplet.name + '_' + block.name)
        return names

    def _validate_steady_heatmap_index(self, sequence_steps):
        try:
            time_heatmap = float(self.args.time_heatmap)
            time_step = float(self.args.time_step)
        except (AttributeError, TypeError, ValueError):
            return

        if time_step <= 0:
            raise ValueError(f'time_step must be positive, got {time_step}')

        step_index = int(round(time_heatmap/time_step))
        if step_index < 0 or step_index > sequence_steps:
            raise ValueError(
                f'steady time_heatmap selects power sequence index {step_index}, '
                f'outside valid range 0..{sequence_steps}'
            )

    @staticmethod
    def _count_steps(duration, interval, label):
        raw_steps = float(duration) / float(interval)
        steps = int(round(raw_steps))
        if not np.isclose(raw_steps, steps, rtol=1e-9, atol=1e-9):
            raise ValueError(f'{label} must be an integer number of steps, got {raw_steps}')
        return steps

    @staticmethod
    def _preview_names(names, limit=5):
        preview = ', '.join(names[:limit])
        if len(names) > limit:
            preview += ', ...'
        return preview

if __name__ == '__main__':
    power_config_dict = common.load_dict_yaml('power_dist_config.yml')
    grid = Power_grid(power_config_dict, 'power_seq.csv')

    grid.create_power_seq_grid()

    for layer in grid.power_layers:
        if layer.name == 'chiplet_1':
            for node in layer.power_nodes:
                print(node.name, node.power_seq, node.x, node.y, node.length_x, node.length_y)