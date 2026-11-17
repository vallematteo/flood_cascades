# Reproducing the paper

Every figure, table and quoted number in the paper and the Supplementary Information (SI), with
the script that produces it, the data it reads, and how that data was generated.

Figure and table numbers follow the submitted PDFs. All scripts are run from the repository
root and write to `output/`.

Every script runs in the uv environment (`uv run python <script>`); `bash reproduce.sh` runs
them all, in about 12 minutes on an Apple M3 Pro: a few seconds per figure, except SI Figs 4
to 7 (about 80 s each, for the PAWN permutation thresholds), `tables.py` (20 s) and
`paper_numbers.py` (about 5 minutes).

**Checks.** Rendered with the Python environment that produced the paper (conda, matplotlib
3.10.8 with FreeType 2.14.1), every figure script reproduces the paper PNG pixel for pixel,
except SI Fig. 3 (contour labels placed differently: the PNG in the paper predates the last edit
of its script) and the error bars of the Sobol' panels (SALib's bootstrap was unseeded; the
package seeds it). The uv environment uses matplotlib's wheel, which bundles FreeType 2.6.1, so
text and line edges are anti-aliased slightly differently; the content is identical.
`tables.py` reproduces every cell of the result tables except one (SI Table 9, delta = 0.2,
S1: 30.4 against 30.5 in the paper). `paper_numbers.py` recomputes 325 numbers of the paper and
the SI; 318 match, and the 7 others are rounding slips in the text.

## Experiments

The data in `data/` comes from four simulation experiments. Each has its own folder in
`experiments/` with the parameter design, the runner, and the script that reduces raw output
to the files in `data/`.

| Experiment | Purpose | Runs | Where | Cost |
|---|---|---|---|---|
| `abc_calibration` | calibration at the no-flood scenario (rejection ABC) | 10,010 | SLURM cluster | TODO core-hours |
| `global_sa` | global sensitivity analysis (Saltelli design, Sobol' and PAWN indices) | 166,400 | SLURM cluster | TODO core-hours |
| `cascades_rnet` | foreclosure cascades and the net reproduction number, by flood scenario and spillover intensity | 65,520 | SLURM cluster | TODO core-hours |
| `foreclosure_timing` | age and distress history at foreclosure | 1,050 | laptop | about 15 core-hours (8 s per run) |

The raw output of the cluster experiments is not included (TODO: size). Each experiment folder
explains how to regenerate it, and the reduced data it produces is included in `data/`.

## Computing resources

The three cluster experiments ran on the Snellius national supercomputer (SURF, the
Netherlands), partition `fat_genoa` (48 nodes, `fcn[73-120]`): nodes with 2 AMD EPYC 9654
(Genoa) processors, 192 cores and 1,474,560 MB (1.41 TiB) of schedulable memory, with a default
of 7,680 MB per core (SLURM `DefMemPerCPU`; from `scontrol show partition fat_genoa` and
`scontrol show node`). Each experiment ran as one SLURM job on one node
(the global sensitivity analysis as a 5-task array, one node per task), with 192 single-core
Julia workers and 336 GB of memory requested per node (`--ntasks=192 --cpus-per-task=1
--mem=336G`). That node type was the one available; the experiments do not need it.

**Memory.** One Julia worker (one process running simulations one after the other) peaks at
about 1.2 GB: 1.10 GB for a single run, 1.15 GB for 8 consecutive runs of the heaviest cells
(S4, delta = 0.25), measured with `/usr/bin/time -l` on an Apple M3 Pro, Julia 1.10.4, Julia
runtime and compiled code included. A node therefore needs about 1.2 GB per worker plus a few
GB for the main process: 192 workers fit in about 240 GB, and any machine with at least 1.5 GB
per core can run one worker per core. The runner of the sensitivity analysis and the
calibration, which also writes Parquet output, peaks at 0.91 GB per worker (110 runs).

**Time.** One run takes 8 to 9 s on one core, on a Genoa node and on an Apple M3 Pro alike.

| Experiment | SLURM job | Runs | Node time | Core-hours (192 cores x node time) |
|---|---|---|---|---|
| `global_sa` | array of 5 tasks, one node each, 2026-10-28 | 166,400 | about 30 min per task | TODO (`sacct`) |
| `abc_calibration` | one node, 2026-11-04 | 10,010 | TODO | TODO |
| `cascades_rnet` | one node, 2026-11-10 | 65,520 | about 1 h (simulation 55 min: 13 cells of 5,040 runs, 250 to 280 s each, mean 8.7 s per run) plus Julia start-up | TODO (`sacct`) |
| `foreclosure_timing` | laptop, 10 cores, 2026-11-10 | 1,050 | TODO | |

The cascade timings are from the log of an identical design run on 2026-11-05 (job on node
`fcn98`, 13 chunks of 5,040 runs, 3,315 s of simulation in all).

## Main text

| Item | Script | Data | Data from |
|---|---|---|---|
| Fig. 1 | none: schematic of the model, not generated from data | | |
| Fig. 2 | `figure_2.py` | `data/abc_calibration/` | `abc_calibration` |
| Fig. 3a | `figure_3.py` | `data/cascades_rnet/runs_all.csv.gz` | `cascades_rnet` |
| Fig. 3b | `figure_3.py` | `data/global_sa/` | `global_sa` |
| Fig. 4 | `figure_4.py` | `data/cascades_rnet/` (paired seeds, cascade sizes) | `cascades_rnet` |
| Fig. 5 | `figure_5.py` | `data/foreclosure_timing/gap_*` | `foreclosure_timing` |
| Table 1 | `tables.py` | GEV parameters (`utils.py`) | `preparation/gev/` |
| Numbers in the text | `paper_numbers.py` | all of `data/`, `preparation/initialization/baseline_parameters.csv` | all |

## Supplementary Information

| Item | Script | Data | Data from |
|---|---|---|---|
| SI Fig. 1 | `figure_S1.py` | GEV parameters (`utils.py`) | `preparation/gev/` |
| SI Fig. 2 | `figure_S2.py` | `preparation/flood_maps/`, `data/cascades_rnet/fcl_cells.csv.gz`, `runs_all.csv.gz` | `cascades_rnet` |
| SI Fig. 3 | `figure_S3.py` | `data/global_sa/runs_set*.csv.gz` | `global_sa` |
| SI Figs 4 to 6 | `figure_S4.py`, `figure_S5.py`, `figure_S6.py` (panels a, b, c) | `data/global_sa/` | `global_sa` |
| SI Figs 7 and 8 | `figure_S7.py` (panels a, b), `figure_S8.py` | `data/global_sa/` | `global_sa` |
| SI Figs 9 and 10 | `figure_S9.py`, `figure_S10.py` | `data/cascades_rnet/paired_seed.csv.gz` | `cascades_rnet` |
| SI Figs 11 and 12 | `figure_S11.py`, `figure_S12.py` | `data/foreclosure_timing/` | `foreclosure_timing` |
| SI Tables 1 to 4 | none: model specification (parameters, distributions, attributes, scorecard) | | |
| SI Table 5 | TODO: admission of 2 x 10^6 applicant draws | | |
| SI Table 6 | `tables.py` | `data/abc_calibration/` | `abc_calibration` |
| SI Table 7 | none: sampling ranges of the sensitivity analysis | | |
| SI Table 8 | `tables.py` | `data/global_sa/` | `global_sa` |
| SI Tables 9 and 10 | `tables.py` | `data/cascades_rnet/` | `cascades_rnet` |
| Numbers in the SI | `paper_numbers.py` | all of `data/` | all |

## Model inputs

| Input | File | Produced by |
|---|---|---|
| flood-severity series (2,500 months per scenario) | `preparation/gev/gev_config_static_2500.csv` | fixed reference input of every simulation in the paper; new seeded series: `uv run python preparation/gev/generate_gev_series.py` |
| GEV scenario parameters and tier probabilities | `preparation/gev/gev_scenario_parameters.py` | `python preparation/gev/gev_scenario_parameters.py` |
| CBS tables 85887NED (income deciles) and 80590NED (unemployment) | `preparation/initialization/cbs/<table>_<pull date>.csv` | `python preparation/initialization/download_cbs.py` |
| baseline m, ε (with s.e.), exp_lb, η (SI, Baseline parametrisation) | `preparation/initialization/baseline_parameters.csv` | `python preparation/initialization/baseline_parameters.py` |
| calibration targets: NHG foreclosure-rate and BKR distress bands | `preparation/calibration_targets/` | `python preparation/calibration_targets/build_calibration_targets.py` |
