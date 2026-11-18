# Global sensitivity analysis

Saltelli design of 3,328 parameter vectors (11 parameters), 5 flood scenarios, 10 seeds:
166,400 runs. Sobol' first- and total-order indices and PAWN indices of the foreclosure
counts by channel, the share of financial foreclosures and the mean number of distress months.

Used by: Fig. 3b, SI Figs 3 to 8, SI Table 8, and the numbers of the sensitivity analysis.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Design | `design.py` | `output/global_sa/design/` (problem definition, Saltelli matrix, candidate designs; published in `data/global_sa/design/`), `output/global_sa/run_list.csv` |
| 2. Simulation | `run.jl` (`slurm_run.sh`: 5 array tasks of 33,280 runs) | Parquet output per run in `output/global_sa/raw/` (three files per run, 499,200 files) |
| 3. Reduction | `reduce.py` (`slurm_reduce.sh`) | `output/global_sa/reduced/runs_set{0-4}.csv.gz` (per run), `vectors_set{0-4}.csv` (per vector); published in `data/global_sa/` |

Every script writes to `output/global_sa/` (reductions to `output/global_sa/reduced/`) and never to
`data/`, so the published files in `data/global_sa/` stay as the reference to compare a rerun with
(see [REPRODUCE.md](../../REPRODUCE.md#experiments)).

## Run

    python experiments/global_sa/design.py
    sbatch experiments/global_sa/slurm_run.sh
    sbatch --dependency=afterok:<job id> experiments/global_sa/slurm_reduce.sh

The runs of a few parameter vectors on a laptop, for example to check that they reproduce the
shipped data (50 runs per vector, about a minute per vector on 10 cores):

    julia --project=. experiments/global_sa/run.jl --sample-ids 0,1000 --raw-dir output/global_sa_check
    python experiments/global_sa/reduce.py --raw-dir output/global_sa_check --out-dir output/global_sa_check/reduced

## Reproducibility

Checked on 2026-11-17:
- `design.py` regenerates the files of `data/global_sa/design/` byte for byte, and its run list matches
  the 166,400 runs of the cluster job (run ids, scenarios, seeds and parameters).
- The 100 runs of vectors 0 and 1000, rerun on a laptop (Apple M3 Pro, Julia 1.10.4) and
  reduced with `reduce.py`, reproduce their rows of `data/global_sa/`: every count is
  identical, the per-vector statistics are identical, and the per-run rates agree to the last
  binary digit (at most 4e-13; the shipped files were written on the cluster with an older
  pandas).
- Peak memory per worker process: 0.91 GB, Parquet output included.
