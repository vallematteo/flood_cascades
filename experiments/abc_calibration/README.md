# Calibration

Rejection approximate Bayesian computation at the no-flood scenario: 1,001 parameter
combinations (a scrambled Sobol' sequence of 2^13 points on the sensitivity-analysis ranges,
restricted to beta1 - beta2 >= 1/2) x 10 seeds = 10,010 runs, compared with the NHG
foreclosure-rate and BKR distress bands (`preparation/calibration_targets/`).

Used by: Fig. 2, SI Table 6, and the calibration numbers.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Design | `design.py` | `data/abc_calibration/design.csv`, `output/abc_calibration/run_list.csv` |
| 2. Simulation | `experiments/global_sa/run.jl` | Parquet output per run in `output/abc_calibration/raw/` |
| 3. Reduction | `experiments/global_sa/reduce.py --runs-only` | `data/abc_calibration/runs_set0.csv.gz` (per run: foreclosure rate, distress share) |

`slurm_run.sh` runs steps 2 and 3 in one job on a cluster node.

## Run

    uv run python experiments/abc_calibration/design.py
    sbatch experiments/abc_calibration/slurm_run.sh

The 10 runs of one combination on a laptop, for example the calibrated combination 226:

    julia --project=. experiments/global_sa/run.jl --run-list output/abc_calibration/run_list.csv --sample-ids 226 --raw-dir output/abc_check
    uv run python experiments/global_sa/reduce.py --run-list output/abc_calibration/run_list.csv --raw-dir output/abc_check --out-dir output/abc_check/reduced --runs-only

## Reproducibility

Checked on 2026-11-17: `design.py` reproduces the 1,001 combinations and the run list of the
cluster job exactly; the 10 runs of combination 226, rerun on a laptop and reduced, reproduce
their rows of `runs_set0.csv.gz` (identical counts; rates equal to the last binary digit).
