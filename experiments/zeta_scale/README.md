# Scale of initial liquid funds

Test of `zeta_scale`, the factor applied to the fitted ratio of liquid funds to the loan
(0.015 in every other simulation, a keyword argument of the model). 101 values from 0.005 to
0.03 in steps of 0.00025 x 5 seeds = 505 runs, at the no-flood scenario with delta = 0, at the
calibrated parametrisation and the baseline values of the other parameters.

Used by: the zeta_scale numbers of the SI section on agent initialisation (foreclosure rate and
distress share between scales 0.008 and 0.02, median loan age at financial foreclosure,
admission rate and mean population at scales 0.01, 0.015 and 0.02).

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Design | `design.py` | `output/zeta_scale/run_list.csv` |
| 2. Simulation | `experiments/global_sa/run.jl` | Parquet output per run in `output/zeta_scale/raw/` |
| 3. Reduction | `reduce.py` | `data/zeta_scale/runs.csv.gz` (per run: admission rate, mean population, foreclosure rate, distress share, median age at financial foreclosure) |

## Run

On a laptop (about 7 minutes on 8 cores):

    uv run python experiments/zeta_scale/design.py
    julia --project=. experiments/global_sa/run.jl --run-list output/zeta_scale/run_list.csv --raw-dir output/zeta_scale/raw --num-workers 8
    uv run python experiments/zeta_scale/reduce.py

## Reproducibility

Checked on 2026-11-17: the 505 runs, rerun with the model of this repository, reproduce the
original runs of 2026-11-16 exactly (identical foreclosure and distress counts, admission
rates, populations and median ages in every run).

The median loan ages quoted in the SI (51, 61 and 69 months at scales 0.01, 0.015 and 0.02) are
the mean over the five seeds of the per-run median.
