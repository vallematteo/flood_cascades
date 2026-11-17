# Model and calibration inputs

Everything the model and the calibration read. The model needs `flood_maps/` and `gev/`; the
calibration needs `calibration/`.

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

`gev_config_static_2500.csv` is the reference file: the paper's results require this exact
series. TODO: the script that draws a new series from the scenario parameters.

## `calibration/`

| File | Content |
|---|---|
| `85887NED.csv` | CBS StatLine table 85887NED (income and expenditure by income decile), used for the expenditure-share fit of m and ε |
| `83679NED.csv` | CBS StatLine table 83679NED (expenditure shares by income decile), used for the lower bound on non-housing expenditure |
| `targets/` | the empirical bands for the calibration (NHG foreclosure rate, BKR financial distress), every source figure behind them, and the script that builds them; see [targets/README.md](targets/README.md) |

CBS tables are published under CC-BY 4.0. TODO: pull dates.
