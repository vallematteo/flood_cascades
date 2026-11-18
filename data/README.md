# Reduced simulation output

The only data the figure, table and number scripts read. Each folder holds the reduced
output of one experiment; the raw output of the simulations is not included, and each
experiment folder in `experiments/` contains the scripts that regenerate it and reduce it
to these files.

No script writes to this folder. The scripts in the "Produced by" columns write to
`output/<experiment>/reduced/` (the designs to `output/<experiment>/`), under the same file
names, so a rerun can be compared with these published files.

`.csv.gz` files are gzip-compressed CSV; pandas reads them directly (`pd.read_csv`).

**`setting` is the flood scenario** in every file (and in the code): 0 to 4 for S0 (no floods)
to S4. The flood-severity series `preparation/gev/gev_config_static_2500.csv` names the same
scenarios `set0` to `set4`.

## `abc_calibration/`

Calibration at the no-flood scenario (rejection ABC). Run on a SLURM cluster, 2026-11-04.

| File | Content | Produced by |
|---|---|---|
| `design.csv` | the 1,001 parameter combinations (one row per `sample_id`): the points of a scrambled Sobol' sequence of 2^13 points (`raw_index`) on the sensitivity-analysis ranges with beta1 - beta2 >= 1/2 | `experiments/abc_calibration/design.py` |
| `runs_set0.csv.gz` | one row per run (combination x 10 seeds): parameters, trigger and foreclosure counts by channel, mean active mortgages, `rate_tot` (annual foreclosures per 100 active mortgages) and `arrears_fin` (annual share of households in financial distress, %) | `experiments/global_sa/reduce.py --runs-only` |
| `calibrated/runs_set0.csv.gz` | the same per-run table for the calibrated parametrisation at the baseline values of the other parameters (beta1 = 1, beta2 = 0.5, beta3 = 1, kappa = 10), S0, delta = 0, 20 seeds of the cascade experiment (run on a laptop, 2026-11-17): the foreclosure rate (0.42%) and the share of households in financial distress (0.51%) of the main text | `experiments/global_sa/reduce.py --runs-only` on `output/abc_calibration/run_list_calibrated.csv` |

## `global_sa/`

Global sensitivity analysis: Saltelli design of 3,328 parameter vectors x 5 flood scenarios
x 10 seeds (166,400 runs). Run on a SLURM cluster, 2026-10-28 to 2026-10-29.

| File | Content | Produced by |
|---|---|---|
| `design/problem_definition.csv` | parameter names and sampling ranges | `experiments/global_sa/design.py` |
| `design/saltelli_matrix.csv` | the 3,328 parameter vectors, by `saltelli_sample_id` | `experiments/global_sa/design.py` |
| `design/design_config.csv` | the five candidate designs, their centered L2-discrepancy, the chosen sampling seed | `experiments/global_sa/design.py` |
| `runs_set{0-4}.csv.gz` | one row per run in scenario 0 to 4: parameters, trigger and foreclosure counts by channel, mean active mortgages, annual foreclosure rates and distress shares | `experiments/global_sa/reduce.py` |
| `vectors_set{0-4}.csv` | one row per parameter vector in scenario 0 to 4, seeds pooled: foreclosures per run by channel (`n_financial`, `n_strategic`, `n_total`), share of financial foreclosures (`perc_fin`), mean distress months per household (`mean_months`) | `experiments/global_sa/reduce.py` |

## `cascades_rnet/`

Foreclosure cascades and the net reproduction number: calibrated parametrisation, flood
scenarios S0 to S4, spillover intensity delta from 0 to 0.25, 20 seeds (65,520 runs). Run on a
SLURM cluster, 2026-11-10.

| File | Content | Produced by |
|---|---|---|
| `runs_all.csv.gz` | one row per run: design block, scenario, lambda, kappa, beta2, vartheta, delta, seed, foreclosure counts and rates by channel | `experiments/cascades_rnet/reduce_runs.py` |
| `paired_seed.csv.gz` | per seed: flooded runs against the S0 run of the same seed | `experiments/cascades_rnet/reduce_runs.py` |
| `sizes_by_run.csv.gz`, `trees_by_run.csv.gz` | foreclosure cascades reconstructed as weighted trees: cascade sizes per run, and tree counts per run | `experiments/cascades_rnet/reconstruct_cascades.py` |
| `af_runs.csv`, `af_sizes_by_run.csv` | attribution-free spatio-temporal clusters of foreclosures, per run and their sizes | `experiments/cascades_rnet/cluster_attribution_free.py` |
| `fcl_cells.csv.gz` | post-warm-up foreclosures of the delta = 0 runs at kappa = 10, beta1 - beta2 = 0.5, vartheta = 0.05 (the maps of SI Fig. 2 and the share of strategic foreclosures in undamaged homes), per run, map cell, channel and flood status: `run_id, x, y, channel` (1 financial, 0 strategic), `flooded` (1 if the home was damaged in the month of foreclosure), `n_fcl` | `experiments/cascades_rnet/reduce_fcl_cells.py` |

## `foreclosure_timing/`

Age and distress history at foreclosure, calibrated parametrisation. Run on a laptop,
2026-11-10. Two designs: `gap` (beta1 - beta2 from 0.45 to 0.7, 750 runs) and `beta3`
(beta3 from 0.5 to 2, 300 runs).

| File | Content |
|---|---|
| `{gap,beta3}_runs.csv` | one row per run: scenario, seed, beta1, beta2, beta3, lambda, delta, eta, m, varepsilon, foreclosure and trigger counts by channel, mean active population, warm-up length |
| `{gap,beta3}_foreclosures.csv.gz` | one row per foreclosure after the warm-up: `run_id`, `step` (model month), `t` (mortgage age, months), `id` (household), `default_reason` (1 financial, -1 strategic), `ltv`, `n_dis` (distress months including the foreclosure), `n_dis_fin`, `n_dis_strat`, `t_dis_first` (mortgage age at the first distress month, -1 if none) |

Produced by `experiments/foreclosure_timing/reduce_foreclosure_timing.py` from the runner output.

## `zeta_scale/`

Test of the scale of initial liquid funds (`zeta_scale`, 0.015 in every other simulation): 101
values from 0.005 to 0.03 x 5 seeds (505 runs), S0, delta = 0, calibrated parametrisation.
Run on a laptop, 2026-11-17.

| File | Content | Produced by |
|---|---|---|
| `runs.csv.gz` | one row per run: `zeta_scale`, `seed_index`, `seed`, `acceptance` (share of applicants admitted), `mean_active`, foreclosure counts by channel, `n_hhyr_fin`, `rate_tot` (annual foreclosures per 100 active mortgages), `arrears_fin` (annual share of households in financial distress, %), `median_age_fin` (median loan age at financial foreclosure, months) | `experiments/zeta_scale/reduce.py` |
