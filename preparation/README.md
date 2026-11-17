# Model inputs and their preparation

| Folder | Used by |
|---|---|
| `flood_maps/`, `gev/` | every simulation |
| `initialization/` | the baseline parametrisation: initial parameter values and the centre of the sensitivity-analysis ranges |
| `calibration_targets/` | the calibration (ABC) at the no-flood scenario |

## `flood_maps/`

Rasters on the model grid, read by `model/utils.jl` (`read_map`).

| File | Content |
|---|---|
| `grote.png` | flood extent and depth, frequent floods (minor damage tier) |
| `middel.png` | medium-probability floods (moderate tier) |
| `kleine.png` | rare floods (severe tier) |
| `extreme_kleine.png` | extremely rare floods (catastrophic tier) |
| `nvm_cleaned.png` | housing density |

Source: Dutch national flood risk maps (Overstromingsrisicokaarten, IPO 2019), preprocessed to the
model grid as described in the SI. TODO: housing-density source, preprocessing script, licence.

## `gev/`

| File | Content |
|---|---|
| `gev_config_static_2500.csv` | the flood-severity series used in every simulation of the paper: one column per scenario (`set0` to `set4`), one row per month, 2,500 months. Month *t* of a run reads row *t*, so all runs and seeds share the same flood sequence. |
| `gev_scenario_parameters.py` | the GEV parameters of the four flood scenarios and the tier probabilities in Table 1 |
| `generate_gev_series.py` | draws a new series of the same form from a seeded generator (written to `output/gev/`) |

`gev_config_static_2500.csv` is the reference file: every simulation of the paper uses this
exact series. It was drawn once from the scenario distributions with an unseeded random state,
so it cannot be regenerated; `generate_gev_series.py` produces new, reproducible series for
simulations with another flood history and never overwrites the reference.

## `initialization/`

| File | Content |
|---|---|
| `download_cbs.py` | downloads the CBS StatLine tables below through the CBS open-data API and saves each pull under its date |
| `cbs/85887NED_2026-03-16.csv` | CBS 85887NED, income and consumption expenditure by income decile; the pull used in the paper (a pull on 2026-11-17 was identical) |
| `cbs/80590NED_2026-11-17.csv` | CBS 80590NED, labour force and unemployment, monthly |
| `baseline_parameters.py` | baseline values of m and ε (nonlinear least squares on the income deciles, with standard errors and the linearised OLS check), exp_lb (mean social assistance norm 2021 to 2023) and η (mean monthly unemployment rate 2021 to 2025) |
| `baseline_parameters.csv` | the output of `baseline_parameters.py` |

The paper's value of η (3.77%) was computed from a pull of 80590NED on 2026-03-17 that was not
saved; the pull of 2026-11-17 gives the same value. CBS tables are published under CC-BY 4.0.

## `calibration_targets/`

The empirical bands for the calibration (NHG foreclosure rate, BKR financial distress), every
source figure behind them, and the script that builds them; see
[calibration_targets/README.md](calibration_targets/README.md).
