# Small Cluster

## What It Shows

This folder is the default small thermal-model configuration used by
`thermal_RC.py` when no command-line paths are supplied. It is not one of the
numbered run-script examples, but it is useful as a compact reference system for
debugging and for DSS experiments.

Run from the parent `thermal_model` directory with:

```bash
python thermal_RC.py
```

That command uses the default paths declared in `thermal_RC.py`.

## Inputs

- `geometry_small_cluster.yml`: compact 9-layer package geometry.
- `power_dist_small_cluster.yml`: mixed IO and standard chiplet-style power
  blocks with per-block maximum power.
- `power_seq_small_cluster.csv`: default power sequence.
- `dss_class.py`: helper code for using the DSS matrices.

## Artifacts

- `disc_A_matrix.csv` and `disc_B_matrix.csv`: existing 406 x 406 precomputed
  discrete state-space matrices.
- A fresh default `python thermal_RC.py` run should create the standard
  `floorplan/`, `heatmaps/`, and `output/` directories under this folder.
- Expected run-generated `output/` files include `power_all.csv`,
  `temperature_all_0.1.csv`, and DSS matrices under `output/`.

## How It Differs

This is the default/debug system. It is smaller than `mid_size_system`, much
smaller than `100_chiplets_homogeneous`, and less focused on teaching a single
geometry concept than Examples 1, 2, and 3.
