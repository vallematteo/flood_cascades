# Reproducing the paper

Every figure, table and quoted number in the paper and the Supplementary Information (SI), with
the script that produces it, the data it reads, and how that data was generated.

Figure and table numbers follow the submitted PDFs. All scripts are run from the repository
root and write to `output/`.

> **Status: work in progress.** A row is completed when its script has been ported and its
> output checked against the figure in the paper. `TODO` marks what is still missing.

## Experiments

The data in `data/` comes from four simulation experiments. Each has its own folder in
`experiments/` with the parameter design, the runner, and the script that reduces raw output
to the files in `data/`.

| Experiment | Purpose | Runs | Where | Cost |
|---|---|---|---|---|
| `abc_calibration` | calibration at the no-flood scenario (rejection ABC) | TODO | SLURM cluster | TODO core-hours |
| `global_sa` | global sensitivity analysis (Saltelli design, Sobol' and PAWN indices) | 166,400 | SLURM cluster | TODO core-hours |
| `cascades_rnet` | foreclosure cascades and the net reproduction number, by flood scenario and spillover intensity | 65,520 | SLURM cluster | TODO core-hours |
| `foreclosure_timing` | age and distress history at foreclosure | 750 | laptop | TODO (median 7.9 s per run) |

The raw output of the cluster experiments is not included (TODO: size). Each experiment folder
explains how to regenerate it, and the reduced data it produces is included in `data/`.

## Main text

| Item | Script | Data | Run time | Data from |
|---|---|---|---|---|
| Fig. 1 | none: schematic of the model, not generated from data | | | |
| Fig. 2 | TODO `figure_2.py` | TODO `data/abc_calibration/` | TODO | `abc_calibration` |
| Fig. 3a | TODO `figure_3.py` | TODO `data/cascades_rnet/` | TODO | `cascades_rnet` |
| Fig. 3b | TODO `figure_3.py` | TODO `data/global_sa/` | TODO | `global_sa` |
| Fig. 4 | TODO `figure_4.py` | TODO `data/cascades_rnet/` | TODO | `cascades_rnet` |
| Fig. 5 | TODO `figure_5.py` | TODO `data/foreclosure_timing/` | TODO | `foreclosure_timing` |
| Table 1 | TODO `tables.py` | `preparation/gev/` | TODO | GEV scenario parameters |
| Numbers in the text | TODO `paper_numbers.py` | TODO | TODO | all |

## Supplementary Information

| Item | Script | Data | Run time | Data from |
|---|---|---|---|---|
| SI Fig. 1 | TODO `figure_S1.py` | `preparation/gev/` | TODO | GEV scenario parameters |
| SI Fig. 2 | TODO `figure_S2.py` | TODO `preparation/flood_maps/`, `data/cascades_rnet/` | TODO | `cascades_rnet` |
| SI Fig. 3 | TODO `figure_S3.py` | TODO `data/global_sa/` | TODO | `global_sa` |
| SI Fig. 4 | TODO `figure_S4.py` | TODO `data/global_sa/` | TODO | `global_sa` |
| SI Fig. 5 | TODO `figure_S5.py` | TODO `data/global_sa/` | TODO | `global_sa` |
| SI Fig. 6 | TODO `figure_S6.py` | TODO `data/global_sa/` | TODO | `global_sa` |
| SI Fig. 7 | TODO `figure_S7.py` | TODO `data/global_sa/` | TODO | `global_sa` |
| SI Fig. 8 | TODO `figure_S8.py` | TODO `data/global_sa/` | TODO | `global_sa` |
| SI Fig. 9 | TODO `figure_S9.py` | TODO `data/cascades_rnet/` | TODO | `cascades_rnet` |
| SI Fig. 10 | TODO `figure_S10.py` | TODO `data/cascades_rnet/` | TODO | `cascades_rnet` |
| SI Fig. 11 | TODO `figure_S11.py` | TODO `data/foreclosure_timing/` | TODO | `foreclosure_timing` |
| SI Fig. 12 | TODO `figure_S12.py` | TODO `data/foreclosure_timing/` | TODO | `foreclosure_timing` |
| SI Tables 1–5 | none: model specification (parameters, distributions, attributes, scorecard) | TODO: confirm SI Table 5 | | |
| SI Table 6 | TODO `tables.py` | TODO `data/abc_calibration/` | TODO | `abc_calibration` |
| SI Table 7 | none: sampling ranges of the sensitivity analysis | | | |
| SI Table 8 | TODO `tables.py` | TODO `data/global_sa/` | TODO | `global_sa` |
| SI Tables 9–10 | TODO `tables.py` | TODO `data/cascades_rnet/` | TODO | `cascades_rnet` |
| Numbers in the SI | TODO `paper_numbers.py` | TODO | TODO | all |

## Model inputs

| Input | File | Produced by |
|---|---|---|
| flood-severity series (2,500 months per scenario) | `preparation/gev/gev_config_static_2500.csv` | fixed reference input of every simulation in the paper; new seeded series: `uv run python preparation/gev/generate_gev_series.py` |
| GEV scenario parameters and tier probabilities | `preparation/gev/gev_scenario_parameters.py` | `python preparation/gev/gev_scenario_parameters.py` |
| CBS tables 85887NED (income deciles) and 80590NED (unemployment) | `preparation/initialization/cbs/<table>_<pull date>.csv` | `python preparation/initialization/download_cbs.py` |
| baseline m, ε (with s.e.), exp_lb, η (SI, Baseline parametrisation) | `preparation/initialization/baseline_parameters.csv` | `python preparation/initialization/baseline_parameters.py` |
| calibration targets: NHG foreclosure-rate and BKR distress bands | `preparation/calibration_targets/` | `python preparation/calibration_targets/build_calibration_targets.py` |
