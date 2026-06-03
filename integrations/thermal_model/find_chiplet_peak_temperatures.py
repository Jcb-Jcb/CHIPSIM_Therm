#!/usr/bin/env python3
"""Find peak temperature and physical node location for each chiplet."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import common


@dataclass
class NodeRect:
    local_node_id: int
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float


@dataclass
class NodeInfo(NodeRect):
    chiplet: str
    global_node_id: int


@dataclass
class PowerChiplet:
    name: str
    chiplet_x: float
    chiplet_y: float
    length_chiplet_x: Optional[float] = None
    length_chiplet_y: Optional[float] = None
    nodes_x: Optional[int] = None
    nodes_y: Optional[int] = None


@dataclass
class PeakRecord:
    example: str
    layer: str
    chiplet: str
    peak_k: Optional[float] = None
    time_s: Optional[float] = None
    global_node_id: Optional[int] = None
    local_node_id: Optional[int] = None
    x_mm: Optional[float] = None
    y_mm: Optional[float] = None
    center_x_mm: Optional[float] = None
    center_y_mm: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None


def load_yaml(path: Path) -> dict:
    return common.load_dict_yaml(str(path))


def infer_is_homogeneous(power_config: dict) -> bool:
    for layer_config in power_config.values():
        for chiplet_config in layer_config.values():
            if "length_chiplet_x" in chiplet_config or "nodes_x" in chiplet_config:
                return False
    return True


def natural_key(value: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def find_matching_file(patterns: list[str], example_dir: Path) -> Optional[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(example_dir.glob(pattern)))
    matches = [path for path in matches if path.is_file()]
    return matches[0] if matches else None


def find_one(patterns: list[str], example_dir: Path, description: str) -> Path:
    match = find_matching_file(patterns, example_dir)
    if match is None:
        raise FileNotFoundError(f"No {description} found in {example_dir}")
    return match


def find_thermal_log_command_arg(example_dir: Path, flag: str) -> Optional[Path]:
    log_path = example_dir / "thermal.log"
    if not log_path.is_file():
        return None

    working_dir = example_dir
    command = None
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("Working directory:"):
                working_dir = Path(line.split(":", 1)[1].strip())
            elif line.startswith("Command:"):
                command = line.split(":", 1)[1].strip()
                break

    if command is None:
        return None

    tokens = shlex.split(command)
    if flag not in tokens:
        return None
    arg_index = tokens.index(flag) + 1
    if arg_index >= len(tokens):
        return None

    path = Path(tokens[arg_index])
    if not path.is_absolute():
        path = working_dir / path
    return path if path.is_file() else None


def load_thermal_metadata(example_dir: Path) -> dict:
    metadata_path = example_dir / "thermal_adapter_metadata.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open() as f:
        return json.load(f)


def metadata_path(metadata: dict, key: str) -> Optional[Path]:
    value = metadata.get(key)
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def find_geometry_file(example_dir: Path) -> Path:
    local_geometry = find_matching_file(
        [
            "system_geometry.yaml",
            "geometry*.yml",
            "geometry*.yaml",
            "*geometry*.yml",
            "*geometry*.yaml",
        ],
        example_dir,
    )
    if local_geometry is not None:
        return local_geometry

    thermal_log_geometry = find_thermal_log_command_arg(example_dir, "--geometry_file")
    if thermal_log_geometry is not None:
        return thermal_log_geometry

    raise FileNotFoundError(f"No geometry file found in {example_dir}")


def find_power_config_file(example_dir: Path) -> Path:
    local_power_config = find_matching_file(
        [
            "power_config_resolved.yml",
            "power_config_resolved.yaml",
            "power_dist_config*.yml",
            "power_dist*.yml",
            "power_dist*.yaml",
        ],
        example_dir,
    )
    if local_power_config is not None and "chipsim_total" not in local_power_config.name:
        return local_power_config

    metadata_power_config = metadata_path(load_thermal_metadata(example_dir), "resolved_power_config_file")
    if metadata_power_config is not None:
        return metadata_power_config

    thermal_log_power_config = find_thermal_log_command_arg(example_dir, "--power_config_file")
    if thermal_log_power_config is not None:
        return thermal_log_power_config

    raise FileNotFoundError(f"No power config found in {example_dir}")


def find_temperature_file(example_dir: Path) -> Path:
    return find_one(["output/temperature_all_*.csv"], example_dir, "temperature CSV")


def temperature_step_s(temperature_file: Path) -> float:
    match = re.search(r"temperature_all_([0-9.]+)\.csv$", temperature_file.name)
    if not match:
        raise ValueError(f"Could not infer timestep from {temperature_file}")
    return float(match.group(1))


def find_power_layer_name(geometry: dict, power_config: dict) -> str:
    for layer_name, layer_config in geometry["layers"].items():
        if layer_config.get("power_src") and layer_name in power_config:
            return layer_name
    for layer_name in power_config:
        if layer_name in geometry["layers"]:
            return layer_name
    raise ValueError("Could not match a power layer from geometry and power config")


def build_power_layers(power_config: dict) -> dict[str, list[PowerChiplet]]:
    power_layers: dict[str, list[PowerChiplet]] = {}
    for layer_name, layer_config in power_config.items():
        chiplets = [
            PowerChiplet(
                name=chiplet_name,
                chiplet_x=chiplet_config["start_chiplet_x"],
                chiplet_y=chiplet_config["start_chiplet_y"],
                length_chiplet_x=chiplet_config.get("length_chiplet_x"),
                length_chiplet_y=chiplet_config.get("length_chiplet_y"),
                nodes_x=chiplet_config.get("nodes_x"),
                nodes_y=chiplet_config.get("nodes_y"),
            )
            for chiplet_name, chiplet_config in layer_config.items()
        ]
        # Match the simulator's Power_layer sorting: first by y, then stable-sort by x.
        chiplets = sorted(sorted(chiplets, key=lambda c: c.chiplet_y), key=lambda c: c.chiplet_x)
        power_layers[layer_name] = chiplets
    return power_layers


def calculate_length_for_non_uniform_node(
    index: int,
    nodes: list[float],
    start_coordinate: float,
    end_coordinate: float,
) -> tuple[float, float]:
    if index == 0:
        length = nodes[index] + ((nodes[index] + nodes[index + 1]) / 2.0 - nodes[index])
        start_point = start_coordinate
    elif index == len(nodes) - 1:
        length = (
            end_coordinate
            - (nodes[index] + start_coordinate)
            + (nodes[index] - (nodes[index] + nodes[index - 1]) / 2.0)
        )
        start_point = (nodes[index] + nodes[index - 1]) / 2.0 + start_coordinate
    else:
        length = (nodes[index] + nodes[index + 1]) / 2.0 - (
            nodes[index] + nodes[index - 1]
        ) / 2.0
        start_point = (nodes[index] + nodes[index - 1]) / 2.0 + start_coordinate

    return length, start_point


def flatten_grid(grid: list[list[Optional[NodeRect]]]) -> list[NodeRect]:
    nodes: list[NodeRect] = []
    local_id = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            node = grid[i][j]
            if node is None:
                raise ValueError(f"Missing generated node at grid index ({i}, {j})")
            node.local_node_id = local_id
            nodes.append(node)
            local_id += 1
    return nodes


def build_regular_layer_nodes(layer_config: dict, utils: common.Utils) -> list[NodeRect]:
    nodes_config = layer_config["nodes"]
    start = layer_config["start_point"]

    if nodes_config["uniform"]:
        total_x = nodes_config["x_nodes"]
        total_y = nodes_config["y_nodes"]
        x_length = (utils.package_x_len - 2 * start["x"]) / total_x
        y_length = (utils.package_y_len - 2 * start["y"]) / total_y
        return [
            NodeRect(
                local_node_id=i * total_y + j,
                x_mm=i * x_length + start["x"],
                y_mm=j * y_length + start["y"],
                width_mm=x_length,
                height_mm=y_length,
            )
            for i in range(total_x)
            for j in range(total_y)
        ]

    x_nodes = nodes_config["x_nodes"]
    y_nodes = nodes_config["y_nodes"]
    total_y = len(y_nodes)
    layer_nodes: list[NodeRect] = []
    for i in range(len(x_nodes)):
        x_length, x = calculate_length_for_non_uniform_node(
            i, x_nodes, start["x"], utils.package_x_len - start["x"]
        )
        for j in range(len(y_nodes)):
            y_length, y = calculate_length_for_non_uniform_node(
                j, y_nodes, start["y"], utils.package_y_len - start["y"]
            )
            layer_nodes.append(
                NodeRect(
                    local_node_id=i * total_y + j,
                    x_mm=x,
                    y_mm=y,
                    width_mm=x_length,
                    height_mm=y_length,
                )
            )
    return layer_nodes


def build_homogeneous_chiplet_layer_nodes(layer_config: dict, utils: common.Utils) -> list[NodeRect]:
    nodes_config = layer_config["nodes"]
    start = layer_config["start_point"]

    if nodes_config["uniform"]:
        x_nodes_chiplet = nodes_config["x_nodes"]
        y_nodes_chiplet = nodes_config["y_nodes"]
        total_x = x_nodes_chiplet * utils.num_chiplet_x
        total_y = y_nodes_chiplet * utils.num_chiplet_y
        grid: list[list[Optional[NodeRect]]] = [[None for _ in range(total_y)] for _ in range(total_x)]

        x_end = None
        for i_chiplet in range(utils.num_chiplet_x):
            x_start = start["x"] if i_chiplet == 0 else x_end + utils.chiplet_spacing
            x_end = x_start + utils.chiplet_x_len

            y_end = None
            for j_chiplet in range(utils.num_chiplet_y):
                y_start = start["y"] if j_chiplet == 0 else y_end + utils.chiplet_spacing
                y_end = y_start + utils.chiplet_y_len

                x_length = (x_end - x_start) / x_nodes_chiplet
                y_length = (y_end - y_start) / y_nodes_chiplet
                for i in range(x_nodes_chiplet):
                    for j in range(y_nodes_chiplet):
                        grid[i_chiplet * x_nodes_chiplet + i][j_chiplet * y_nodes_chiplet + j] = NodeRect(
                            local_node_id=-1,
                            x_mm=i * x_length + x_start,
                            y_mm=j * y_length + y_start,
                            width_mm=x_length,
                            height_mm=y_length,
                        )

        return flatten_grid(grid)

    x_nodes = nodes_config["x_nodes"]
    y_nodes = nodes_config["y_nodes"]
    total_x = len(x_nodes) * utils.num_chiplet_x
    total_y = len(y_nodes) * utils.num_chiplet_y
    grid: list[list[Optional[NodeRect]]] = [[None for _ in range(total_y)] for _ in range(total_x)]

    x_end = None
    for i_chiplet in range(utils.num_chiplet_x):
        x_start = start["x"] if i_chiplet == 0 else x_end + utils.chiplet_spacing
        x_end = x_start + utils.chiplet_x_len

        y_end = None
        for j_chiplet in range(utils.num_chiplet_y):
            y_start = start["y"] if j_chiplet == 0 else y_end + utils.chiplet_spacing
            y_end = y_start + utils.chiplet_y_len

            for i in range(len(x_nodes)):
                x_length, x = calculate_length_for_non_uniform_node(i, x_nodes, x_start, x_end)
                for j in range(len(y_nodes)):
                    y_length, y = calculate_length_for_non_uniform_node(j, y_nodes, y_start, y_end)
                    grid[i_chiplet * len(x_nodes) + i][j_chiplet * len(y_nodes) + j] = NodeRect(
                        local_node_id=-1,
                        x_mm=x,
                        y_mm=y,
                        width_mm=x_length,
                        height_mm=y_length,
                    )

    return flatten_grid(grid)


def build_heterogeneous_chiplet_layer_nodes(chiplets: list[PowerChiplet]) -> list[NodeRect]:
    nodes: list[NodeRect] = []
    local_id = 0
    for chiplet in chiplets:
        if chiplet.nodes_x is None or chiplet.nodes_y is None:
            raise ValueError(f"Missing node counts for heterogeneous chiplet {chiplet.name}")
        if chiplet.length_chiplet_x is None or chiplet.length_chiplet_y is None:
            raise ValueError(f"Missing dimensions for heterogeneous chiplet {chiplet.name}")
        x_length = chiplet.length_chiplet_x / chiplet.nodes_x
        y_length = chiplet.length_chiplet_y / chiplet.nodes_y
        for i in range(chiplet.nodes_x):
            for j in range(chiplet.nodes_y):
                nodes.append(
                    NodeRect(
                        local_node_id=local_id,
                        x_mm=chiplet.chiplet_x + i * x_length,
                        y_mm=chiplet.chiplet_y + j * y_length,
                        width_mm=x_length,
                        height_mm=y_length,
                    )
                )
                local_id += 1
    return nodes


def build_layer_nodes(
    layer_name: str,
    layer_config: dict,
    utils: common.Utils,
    power_layers: dict[str, list[PowerChiplet]],
    is_homogeneous: bool,
) -> list[NodeRect]:
    if not layer_config["nodes"]["under_chiplet"]:
        return build_regular_layer_nodes(layer_config, utils)
    if is_homogeneous:
        return build_homogeneous_chiplet_layer_nodes(layer_config, utils)
    if layer_name not in power_layers:
        raise ValueError(
            f"Layer {layer_name} is under-chiplet in heterogeneous mode but has no power-config geometry"
        )
    return build_heterogeneous_chiplet_layer_nodes(power_layers[layer_name])


def build_target_layer_nodes(
    geometry: dict,
    power_layers: dict[str, list[PowerChiplet]],
    is_homogeneous: bool,
    target_layer_name: str,
):
    utils = common.Utils(geometry["common"])
    global_start = 0
    target_nodes: Optional[list[NodeRect]] = None
    target_global_start = 0

    for layer_name, layer_config in geometry["layers"].items():
        layer_nodes = build_layer_nodes(
            layer_name=layer_name,
            layer_config=layer_config,
            utils=utils,
            power_layers=power_layers,
            is_homogeneous=is_homogeneous,
        )
        if layer_name == target_layer_name:
            target_nodes = layer_nodes
            target_global_start = global_start
        global_start += len(layer_nodes)

    if target_nodes is None:
        raise ValueError(f"Layer {target_layer_name} was not found in geometry")

    return target_nodes, target_global_start, utils


def overlap_area(node: NodeRect, chiplet_x: float, chiplet_y: float, chiplet_w: float, chiplet_h: float) -> float:
    return common.calculate_overlapping_area(
        x1=node.x_mm,
        y1=node.y_mm,
        x2=chiplet_x,
        y2=chiplet_y,
        x_len1=node.width_mm,
        y_len1=node.height_mm,
        x_len2=chiplet_w,
        y_len2=chiplet_h,
    )


def map_nodes_to_chiplets(
    layer_nodes: list[NodeRect],
    layer_global_start: int,
    chiplets: list[PowerChiplet],
    utils: common.Utils,
    is_homogeneous: bool,
) -> dict[int, NodeInfo]:
    node_map: dict[int, NodeInfo] = {}

    if is_homogeneous:
        for chiplet in chiplets:
            for node in layer_nodes:
                area = overlap_area(
                    node=node,
                    chiplet_x=chiplet.chiplet_x,
                    chiplet_y=chiplet.chiplet_y,
                    chiplet_w=utils.chiplet_x_len,
                    chiplet_h=utils.chiplet_y_len,
                )
                if area <= 0:
                    continue
                global_id = layer_global_start + node.local_node_id
                node_map[global_id] = NodeInfo(
                    local_node_id=node.local_node_id,
                    x_mm=node.x_mm,
                    y_mm=node.y_mm,
                    width_mm=node.width_mm,
                    height_mm=node.height_mm,
                    chiplet=chiplet.name,
                    global_node_id=global_id,
                )
    else:
        flat_nodes = {node.local_node_id: node for node in layer_nodes}
        local_start = 0
        for chiplet in chiplets:
            chiplet_nodes = chiplet.nodes_x * chiplet.nodes_y
            for local_id in range(local_start, local_start + chiplet_nodes):
                node = flat_nodes[local_id]
                global_id = layer_global_start + local_id
                node_map[global_id] = NodeInfo(
                    local_node_id=local_id,
                    x_mm=node.x_mm,
                    y_mm=node.y_mm,
                    width_mm=node.width_mm,
                    height_mm=node.height_mm,
                    chiplet=chiplet.name,
                    global_node_id=global_id,
                )
            local_start += chiplet_nodes

    return node_map


def scan_temperature_file(
    example_name: str,
    layer_name: str,
    temperature_file: Path,
    node_map: dict[int, NodeInfo],
) -> list[PeakRecord]:
    dt = temperature_step_s(temperature_file)
    records = {
        node_info.chiplet: PeakRecord(
            example=example_name,
            layer=layer_name,
            chiplet=node_info.chiplet,
        )
        for node_info in node_map.values()
    }

    tracked_columns = sorted(node_map)
    with temperature_file.open(newline="") as f:
        reader = csv.reader(f)
        for row_index, row in enumerate(reader):
            time_s = row_index * dt
            for global_node_id in tracked_columns:
                if global_node_id >= len(row):
                    raise ValueError(
                        f"{temperature_file} has {len(row)} columns, but node {global_node_id} was requested"
                    )
                temperature_k = float(row[global_node_id])
                node_info = node_map[global_node_id]
                record = records[node_info.chiplet]
                if record.peak_k is None or temperature_k > record.peak_k:
                    record.peak_k = temperature_k
                    record.time_s = time_s
                    record.global_node_id = global_node_id
                    record.local_node_id = node_info.local_node_id
                    record.x_mm = node_info.x_mm
                    record.y_mm = node_info.y_mm
                    record.center_x_mm = node_info.x_mm + node_info.width_mm / 2.0
                    record.center_y_mm = node_info.y_mm + node_info.height_mm / 2.0
                    record.width_mm = node_info.width_mm
                    record.height_mm = node_info.height_mm

    return sorted(records.values(), key=lambda record: natural_key(record.chiplet))


def analyze_example(example_dir: Path) -> list[PeakRecord]:
    geometry_file = find_geometry_file(example_dir)
    power_config_file = find_power_config_file(example_dir)
    temperature_file = find_temperature_file(example_dir)

    geometry = load_yaml(geometry_file)
    power_config = load_yaml(power_config_file)
    is_homogeneous = infer_is_homogeneous(power_config)
    target_layer_name = find_power_layer_name(geometry, power_config)
    power_layers = build_power_layers(power_config)

    target_nodes, layer_global_start, utils = build_target_layer_nodes(
        geometry=geometry,
        power_layers=power_layers,
        is_homogeneous=is_homogeneous,
        target_layer_name=target_layer_name,
    )

    node_map = map_nodes_to_chiplets(
        layer_nodes=target_nodes,
        layer_global_start=layer_global_start,
        chiplets=power_layers[target_layer_name],
        utils=utils,
        is_homogeneous=is_homogeneous,
    )

    if not node_map:
        raise ValueError(f"No chiplet nodes mapped for {example_dir}")

    return scan_temperature_file(
        example_name=example_dir.name,
        layer_name=target_layer_name,
        temperature_file=temperature_file,
        node_map=node_map,
    )


def default_example_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "output").is_dir() and list((path / "output").glob("temperature_all_*.csv"))
    )


def print_records(records: list[PeakRecord]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "example",
            "layer",
            "chiplet",
            "peak_K",
            "peak_C",
            "time_s",
            "global_node_id",
            "local_node_id",
            "node_x_mm",
            "node_y_mm",
            "node_center_x_mm",
            "node_center_y_mm",
            "node_width_mm",
            "node_height_mm",
        ]
    )
    for record in records:
        writer.writerow(
            [
                record.example,
                record.layer,
                record.chiplet,
                f"{record.peak_k:.6f}",
                f"{record.peak_k - 273.15:.6f}",
                f"{record.time_s:.9g}",
                record.global_node_id,
                record.local_node_id,
                f"{record.x_mm:.6f}",
                f"{record.y_mm:.6f}",
                f"{record.center_x_mm:.6f}",
                f"{record.center_y_mm:.6f}",
                f"{record.width_mm:.6f}",
                f"{record.height_mm:.6f}",
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find peak temperature and physical node location for each chiplet."
    )
    parser.add_argument(
        "example_dirs",
        nargs="*",
        type=Path,
        help="Example directories to analyze. Defaults to all child directories with output/temperature_all_*.csv.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Thermal model root used when no example directories are supplied.",
    )
    parser.add_argument(
        "--material-prop-file",
        type=Path,
        default=None,
        help="Deprecated; material properties are not needed for peak lookup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    example_dirs = args.example_dirs or default_example_dirs(root)

    all_records: list[PeakRecord] = []
    for example_dir in example_dirs:
        all_records.extend(analyze_example(example_dir.resolve()))

    print_records(all_records)


if __name__ == "__main__":
    main()
