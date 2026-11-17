# Foreclosure timing

Mortgage age and distress history at foreclosure, at the calibrated parametrisation.

Used by: Fig. 5, SI Figs 11 and 12, and the numbers on foreclosure timing.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Simulation (1,050 runs) | `run.jl` | `output/foreclosure_timing/raw/{gap,beta3}_{runs,foreclosures}.csv` |
| 2. Reduction to the variables the paper uses | `reduce_foreclosure_timing.py` | `data/foreclosure_timing/{gap,beta3}_runs.csv`, `{gap,beta3}_foreclosures.csv.gz` |

Designs: `gap` (beta1 - beta2 in {0.45, 0.5, 0.55, 0.6, 0.7}, 750 runs) and `beta3`
(beta3 in {0.5, 2}, 300 runs); scenarios S0 to S4, lambda in {0.25, 0.5, 0.75}, 10 seeds.

## Run

On a laptop, from the repository root:

    julia --project=. experiments/foreclosure_timing/run.jl --num-workers 10
    python experiments/foreclosure_timing/reduce_foreclosure_timing.py

A run takes about 8 s on one core; the full design takes about TODO minutes on 10 cores of an
Apple M3 Pro.

## Reproducibility

Checked on 2026-11-17: 15 runs (gap runs 0, 1, 75, 150, 299, 300, 450, 600, 700, 749; beta3
runs 0, 30, 150, 180, 299), rerun with this runner and the model in `model/`, reproduce the
output used in the paper exactly (every column of the run table and every foreclosure record).
