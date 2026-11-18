# Foreclosure cascades

Flood scenario x spillover intensity at the calibrated parametrisation: the net reproduction
number of the foreclosure feedback (R_net), the foreclosure-rate multiplier, and the
reconstruction of foreclosure cascades.

Used by: Fig. 3a, Fig. 4, SI Figs 2, 9 and 10, SI Tables 9 and 10, and the numbers of the
cascade section.

## Pipeline

| Step | Script | Input | Output |
|---|---|---|---|
| 1. Simulation (65,520 runs) | `run.jl` | model, `preparation/` | `runs_chunk{0-12}.csv`, `fcl_chunk{0-12}.csv` (raw, about 1.4 GB) |
| 2a. Runs and pairs | `reduce_runs.py` | `runs_chunk*.csv` | `runs_all.csv.gz`, `paired_seed.csv.gz` |
| 2b. Cascade reconstruction | `reconstruct_cascades.py` | `runs_chunk*.csv`, `fcl_chunk*.csv` | `sizes_by_run.csv.gz`, `trees_by_run.csv.gz` |
| 2c. Attribution-free clusters | `cluster_attribution_free.py` | `runs_chunk*.csv`, `fcl_chunk*.csv` | `af_runs.csv`, `af_sizes_by_run.csv` |
| 2d. Foreclosures per map cell | `reduce_fcl_cells.py` | `runs_chunk*.csv`, `fcl_chunk*.csv` | `fcl_cells.csv.gz` |

The raw files go to `--out-dir` of `run.jl` (default `output/cascades_rnet/raw/`); the reduced files
of steps 2a to 2d go to `output/cascades_rnet/reduced/`, and their published copies are in
`data/cascades_rnet/`. `slurm_run.sh` runs all steps in one job on a cluster node.

Every script writes to `output/cascades_rnet/` (reductions to `output/cascades_rnet/reduced/`) and never to
`data/`, so the published files in `data/cascades_rnet/` stay as the reference to compare a rerun with
(see [REPRODUCE.md](../../REPRODUCE.md#experiments)).

## Run

Full design, on a SLURM cluster (from the repository root):

    sbatch experiments/cascades_rnet/slurm_run.sh

A few runs on a laptop, for example to check that they reproduce the shipped data:

    julia --project=. experiments/cascades_rnet/run.jl --run-ids 1680,1840,57120,57280 --num-workers 4 --out-dir output/cascades_check

Each run takes about 9 s on one core. The reduction steps take about 4 minutes on a laptop
(Apple M3 Pro) for the full raw output.

## Reproducibility

Checked on 2026-11-17: runs 1680, 1840, 6720, 6880, 57120 and 57280, rerun on a laptop
(Apple M3 Pro, Julia 1.10.4), reproduce the cluster output exactly (every column of the run
table and every foreclosure record). The reduction steps, rerun on the cluster's raw output,
reproduce every file in `data/cascades_rnet/`.
