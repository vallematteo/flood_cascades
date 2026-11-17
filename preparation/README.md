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
| `generate_gev_series.py` | generates the series from its seed (written to `output/gev/`) and checks it against `gev_config_static_2500.csv` |

Every simulation of the paper reads `gev_config_static_2500.csv`. `generate_gev_series.py`
draws it from the scenario distributions with NumPy's MT19937 generator and the seed in the
script, and reproduces the file byte for byte in the locked environment (scipy 1.11.4; other
scipy versions round `genextreme.ppf` differently in the last bits of a few values).

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
