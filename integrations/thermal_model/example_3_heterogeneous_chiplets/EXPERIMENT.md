# Example 3: Heterogeneous Chiplets

## What It Shows

This example demonstrates non-homogeneous chiplet placement. It uses three
chiplets: one larger chiplet and two smaller chiplets. With
`--is_homogeneous false`, chiplet dimensions, node counts, and optional material
overrides come from the power configuration instead of the shared geometry
defaults.

Run from this directory with:

```bash
bash run3.sh
```

The script runs `thermal_RC.py` with `--is_homogeneous false` and
`--total_duration 50`.

## Inputs

- `chiplet_geometry_3_chiplets_uniform_nodes.yml`: common package stack and
  layer definitions.
- `power_dist_config_heterogeneous.yml`: per-chiplet geometry for the chiplet,
  ubump, and tim layers. `chiplet_1` is larger; `chiplet_3` overrides material
  to `interposer`.
- `power_seq_random_3.csv`: 50 one-second power samples for the three chiplet
  power blocks.

## Artifacts

- `floorplan/*.png`: layer node maps for the heterogeneous layout.
- `floorplan/chiplet_power_.png`, `floorplan/ubump_power_.png`, and
  `floorplan/tim_power_.png`: power or placement maps for under-chiplet layers.
- `heatmaps/*_heatmap.png`: per-layer heatmaps.
- `output/power_all.csv`: 50 x 126 node-power matrix in watts.
- `output/temperature_all_0.1.csv`: 501 x 126 temperature matrix from 0 to
  50 seconds at 0.1 second resolution.
- `output/temperature_chiplet_0.1.csv`: three chiplet-average temperature
  traces.
- `output/disc_A_matrix.csv` and `output/disc_B_matrix.csv`: 126 x 126
  discrete state-space matrices.

## How It Differs

This is the heterogeneous geometry example. Example 1 and Example 2 assume a
regular repeated chiplet layout, while this case lets each chiplet define its own
size, node count, and material. It is smaller than the 100-chiplet and mid-size
systems but exercises the non-homogeneous code path.
