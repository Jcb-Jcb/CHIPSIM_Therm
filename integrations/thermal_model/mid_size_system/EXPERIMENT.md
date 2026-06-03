# Mid-Size System

## What It Shows

This is a larger heterogeneous-style system with multiple chiplet classes. The
power configuration defines IO chiplets, standard chiplets, accelerator-like
groups, and other repeated blocks across a mid-size package.

Run from this directory with:

```bash
bash run1.sh
```

The script uses default transient settings: 24 seconds total duration,
1 second power interval, 0.1 second thermal timestep, heatmap generation enabled,
and DSS generation enabled.

## Inputs

- `geometry_mid_size.yml`: 9-layer package geometry with coarser non-chiplet
  layers and under-chiplet layers.
- `power_dist_mid_size.yml`: many chiplet classes with per-class dimensions,
  node counts, and maximum power.
- `power_seq_mid_size.csv`: power sequence for the mid-size chiplet set.
- `dss_class.py`: helper code for consuming precomputed DSS matrices.

## Artifacts

- `disc_A_matrix.csv` and `disc_B_matrix.csv`: existing 580 x 580 precomputed
  discrete state-space matrices.
- A fresh `run1.sh` invocation should also create the standard `floorplan/`,
  `heatmaps/`, and `output/` directories under this folder.
- Expected run-generated `output/` files include `power_all.csv`,
  `temperature_all_0.1.csv`, and DSS matrices under `output/`.

## How It Differs

This is the medium-complexity system between the small examples and the
100-chiplet case. It is more varied than Example 1 and Example 2 because the
chiplets are not a single repeated four-chiplet pattern, but it is not as large
as the 100-chiplet homogeneous stress case.
