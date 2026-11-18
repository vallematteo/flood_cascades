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
S1: 30.4 against 30.5 in the paper). `paper_numbers.py` recomputes 405 numbers of the paper and
the SI (SI Table 5 cell by cell among them); 387 match. Of the others, 7 are rounding slips in
the text, and 11 are euro-valued cells of SI Table 5 that differ by 1 to 3 euros: the paper's
applicant draws were unseeded, the package draws the same number with a fixed seed.

## Demo

The 20 simulations of the calibrated parametrisation without floods (S0, delta = 0, 20 seeds)
and their reduction. They give the annual foreclosure rate (0.42%) and the share of households
in financial distress (0.51%) of the main text, and reproduce
`data/abc_calibration/calibrated/runs_set0.csv.gz`. Run from the repository root with Julia
1.10.4 (with juliaup: `julia +1.10.4` in place of `julia`); other versions give slightly
different runs.

```bash
# 1. the run list of the calibrated parametrisation
uv run python experiments/abc_calibration/design.py

# 2. the 20 simulations (raw output in output/demo/raw/)
julia --project=. experiments/global_sa/run.jl \
    --run-list output/abc_calibration/run_list_calibrated.csv \
    --raw-dir output/demo/raw --num-workers 4

# 3. their reduction to one row per run (output/demo/reduced/runs_set0.csv.gz)
uv run python experiments/global_sa/reduce.py \
    --run-list output/abc_calibration/run_list_calibrated.csv \
    --raw-dir output/demo/raw --out-dir output/demo/reduced --runs-only --workers 4

# 4. the two rates, and the comparison with the reduced data in data/
uv run python -c "
import pandas as pd
new = pd.read_csv('output/demo/reduced/runs_set0.csv.gz')
ref = pd.read_csv('data/abc_calibration/calibrated/runs_set0.csv.gz')
print(f'{len(new)} runs: annual foreclosure rate {new.rate_tot.mean():.2f}%, '
      f'households in financial distress {new.arrears_fin.mean():.2f}%')
cols = ['run_id', 'n_fcl_fin', 'n_fcl_strat', 'n_hhyr_fin']
same = new[cols].equals(ref[cols]) and (new.rate_tot - ref.rate_tot).abs().max() < 1e-12
print('identical to data/' if same else 'differs from data/')
"
```

Expected output of step 4:

```text
20 runs: annual foreclosure rate 0.42%, households in financial distress 0.51%
identical to data/
```

Run time: about 40 seconds with 4 workers on an Apple M3 Pro (one run takes 8 to 9 s on one
core); compiling the Julia packages on the first run adds about 30 seconds.

## Experiments

The data in `data/` comes from five simulation experiments. Each has its own folder in
`experiments/` with the parameter design, the runner, and the script that reduces raw output
to the files in `data/`.

| Experiment | Purpose | Runs | Where | Cost |
|---|---|---|---|---|
| `abc_calibration` | calibration at the no-flood scenario (rejection ABC) | 10,010 | SLURM cluster | 35 core-hours |
| `global_sa` | global sensitivity analysis (Saltelli design, Sobol' and PAWN indices) | 166,400 | SLURM cluster | 473 core-hours, plus 90 for the reduction |
| `cascades_rnet` | foreclosure cascades and the net reproduction number, by flood scenario and spillover intensity | 65,520 | SLURM cluster | 239 core-hours, plus 2 for the reduction |
| `foreclosure_timing` | age and distress history at foreclosure | 1,050 | laptop | 2.4 core-hours (8 to 9 s per run) |
| `zeta_scale` | scale of initial liquid funds: admission, population and no-flood rates | 505 | laptop | about 1 core-hour (7 min on 8 cores) |

The raw output of the simulations is not included: about 1.9 GB for the cascade experiment
(measured), and roughly 2 to 4 MB per run in Parquet for the sensitivity analysis and the
calibration (estimated from check runs; the size grows with the number of households and
distress events), of the order of 0.5 TB and 20 GB. Each experiment folder explains how to
regenerate it, and the reduced data it produces is included in `data/`.

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
calibration, which also writes Parquet output, peaks at 0.91 GB per worker (110 runs). On the
cluster, the whole job (192 workers and the main process) peaked at 148 GiB for the sensitivity
analysis and the calibration and at 154 GiB for the cascade experiment (SLURM `MaxRSS`), about
0.8 GB per worker.

**Time.** One run takes 8 to 9 s on one core, on a Genoa node and on an Apple M3 Pro alike.

From SLURM accounting (`sacct`); core-hours are allocated cores x elapsed time.

| Experiment | SLURM job | Runs | Elapsed | Core-hours |
|---|---|---|---|---|
| `global_sa`, simulation | array of 5 tasks, one node (192 cores) each, 2026-10-29 | 166,400 | 28 to 31 min per task | 473 |
| `global_sa`, reduction | one node (192 cores), 2026-10-29 | | 28 min | 90 |
| `abc_calibration`, simulation and reduction | one node (192 cores), 2026-11-04 | 10,010 | 11 min | 35 |
| `cascades_rnet`, simulation | one node (192 cores), 2026-11-10 | 65,520 | 75 min | 239 |
| `cascades_rnet`, reduction and cascade trees | 24 and 72 cores, 2026-11-10 | | 1 and 2 min | 2 |
| `foreclosure_timing` | laptop, 10 workers, 2026-11-10 | 1,050 | about 15 min | 2.4 (sum of run times) |
| `zeta_scale` | laptop, 8 workers, 2026-11-17 | 505 | 7 min | about 1 |

Snellius ran Red Hat Enterprise Linux 9.8 (kernel 5.14, x86-64), Julia 1.10.4.

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
| SI Fig. 1 | `figure_SI1.py` | GEV parameters (`utils.py`) | `preparation/gev/` |
| SI Fig. 2 | `figure_SI2.py` | `preparation/flood_maps/`, `data/cascades_rnet/fcl_cells.csv.gz`, `runs_all.csv.gz` | `cascades_rnet` |
| SI Fig. 3 | `figure_SI3.py` | `data/global_sa/runs_set*.csv.gz` | `global_sa` |
| SI Figs 4 to 6 | `figure_SI4.py`, `figure_SI5.py`, `figure_SI6.py` (panels a, b, c) | `data/global_sa/` | `global_sa` |
| SI Figs 7 and 8 | `figure_SI7.py` (panels a, b), `figure_SI8.py` | `data/global_sa/` | `global_sa` |
| SI Figs 9 and 10 | `figure_SI9.py`, `figure_SI10.py` | `data/cascades_rnet/paired_seed.csv.gz` | `cascades_rnet` |
| SI Figs 11 and 12 | `figure_SI11.py`, `figure_SI12.py` | `data/foreclosure_timing/` | `foreclosure_timing` |
| SI Tables 1 to 4 | none: model specification (parameters, distributions, attributes, scorecard) | | |
| SI Table 5 | `tables.py` | none: 2 x 10^6 applicant draws from the model's distributions and scorecard (`utils.draw_applicants`, seed 0) | |
| SI Table 6 | `tables.py` | `data/abc_calibration/` | `abc_calibration` |
| SI Table 7 | none: sampling ranges of the sensitivity analysis | | |
| SI Table 8 | `tables.py` | `data/global_sa/` | `global_sa` |
| SI Tables 9 and 10 | `tables.py` | `data/cascades_rnet/` | `cascades_rnet` |
| Numbers in the SI | `paper_numbers.py` | all of `data/` | all |

## Model inputs

| Input | File | Produced by |
|---|---|---|
| flood-severity series (2,500 months per scenario) | `preparation/gev/gev_config_static_2500.csv` | `uv run python preparation/gev/generate_gev_series.py` (from its seed; checks the result against the file) |
| GEV scenario parameters and tier probabilities | `preparation/gev/gev_scenario_parameters.py` | `python preparation/gev/gev_scenario_parameters.py` |
| CBS tables 85887NED (income deciles) and 80590NED (unemployment) | `preparation/initialization/cbs/<table>_<pull date>.csv` | `python preparation/initialization/download_cbs.py` |
| baseline m, ε (with s.e.), exp_lb, η (SI, Baseline parametrisation) | `preparation/initialization/baseline_parameters.csv` | `python preparation/initialization/baseline_parameters.py` |
| calibration targets: NHG foreclosure-rate and BKR distress bands | `preparation/calibration_targets/` | `python preparation/calibration_targets/build_calibration_targets.py` |
