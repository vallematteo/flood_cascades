# Reduced simulation output

The only data the figure, table and number scripts read. Each folder holds the reduced
output of one experiment; the raw output of the simulations is not included, and each
experiment folder in `experiments/` contains the scripts that regenerate it and reduce it
to these files.

`.csv.gz` files are gzip-compressed CSV; pandas reads them directly (`pd.read_csv`).

> TODO: the reduction scripts of `abc_calibration`, `global_sa` and `cascades_rnet` (except
> `fcl_cells`) are being ported into `experiments/`; the paths below name their final location.

## `abc_calibration/`

Calibration at the no-flood scenario (rejection ABC). Run on a SLURM cluster, 2026-11-04.

| File | Content | Produced by |
|---|---|---|
| `design.csv` | the 1,001 parameter combinations (one row per `sample_id`): the points of a scrambled Sobol' sequence of 2^13 points (`raw_index`) on the sensitivity-analysis ranges with beta1 - beta2 >= 1/2 | `experiments/abc_calibration/design.py` (TODO) |
| `runs.csv` | one row per run (combination x 10 seeds): parameters, foreclosure and distress counts, mean active population, `rate_tot` (annual foreclosures per 100 active mortgages) and `arrears_fin` (share of households in financial distress) | the distress reduction (TODO) |

## `global_sa/`

Global sensitivity analysis: Saltelli design of 3,328 parameter vectors x 10 seeds x 5 flood
scenarios (166,400 runs). Run on a SLURM cluster, 2026-10-28 to 2026-10-29.

| File | Content | Produced by |
|---|---|---|
| `design/problem_definition.csv` | parameter names and sampling ranges | design script (TODO) |
| `design/saltelli_matrix.csv` | the 3,328 parameter vectors, by `saltelli_sample_id` | design script (TODO) |
| `counts/statistic_outputs_set{0-4}.csv` | per vector and flood scenario (`set`): foreclosure counts by channel (`n_financial`, `n_strategic`, `n_total`) | count reduction (TODO) |
| `pooled/statistic_outputs_set{0-4}.csv` | per vector, seeds pooled: share of financial foreclosures and distributional statistics | pooled reduction (TODO) |
| `distress/statistic_outputs_set{0-4}.csv`, `distress/distress_outputs_set{0-4}.csv` | per vector: distress and forbearance statistics (mean distress months per household, forbearance shares) | distress reduction (TODO) |
| `distress/rates_runs_set{0-4}.csv.gz` | one row per run: parameters, foreclosure and distress counts, mean active population, annual rates | distress reduction (TODO) |

## `cascades_rnet/`

Foreclosure cascades and the net reproduction number: calibrated parametrisation, flood
scenarios S0 to S4, spillover intensity delta from 0 to 0.25, 20 seeds (65,520 runs). Run on a
SLURM cluster, 2026-11-10.

| File | Content | Produced by |
|---|---|---|
| `runs_all.csv.gz` | one row per run: design block, scenario, lambda, kappa, beta2, vartheta, delta, seed, foreclosure counts and rates by channel | run reduction (TODO) |
| `paired_seed.csv.gz` | per seed: flooded runs against the S0 run of the same seed | run reduction (TODO) |
| `sizes_by_run.csv.gz`, `trees_by_run.csv.gz` | foreclosure cascades reconstructed as weighted trees: cascade sizes per run, and tree counts per run | cascade reconstruction (TODO) |
| `af_runs.csv`, `af_sizes_by_run.csv` | attribution-free spatio-temporal clusters of foreclosures, per run and their sizes | attribution-free clustering (TODO) |
| `fcl_cells.csv.gz` | post-warm-up foreclosures per (run, map cell) for the delta = 0 runs drawn in the maps of SI Fig. 2: `run_id, x, y, n_fcl` | `experiments/cascades_rnet/reduce_fcl_cells.py` |

## `foreclosure_timing/`

Age and distress history at foreclosure, calibrated parametrisation. Run on a laptop,
2026-11-10. Two designs: `gap` (beta1 - beta2 from 0.45 to 0.7, 750 runs) and `beta3`
(beta3 from 0.5 to 2, 300 runs).

| File | Content |
|---|---|
| `{gap,beta3}_runs.csv` | one row per run: scenario, seed, beta1, beta2, beta3, lambda, delta, eta, m, varepsilon, foreclosure and trigger counts by channel, mean active population, warm-up length |
| `{gap,beta3}_foreclosures.csv.gz` | one row per foreclosure after the warm-up: `run_id`, `step` (model month), `t` (mortgage age, months), `id` (household), `default_reason` (1 financial, -1 strategic), `ltv`, `n_dis` (distress months including the foreclosure), `n_dis_fin`, `n_dis_strat`, `t_dis_first` (mortgage age at the first distress month, -1 if none) |

Produced by `experiments/foreclosure_timing/reduce_foreclosure_timing.py` from the runner output.
