# Flood cascades

Code and data for the paper *Neighbourhood flood damage amplifies mortgage default cascades*.

## Contents

- [Quick start](#quick-start)
- [Overview](#overview)
- [Repository contents](#repository-contents)
- [System requirements](#system-requirements)
- [Installation guide](#installation-guide)
- [Demo](#demo)
- [Reproducing the paper](#reproducing-the-paper)
- [License](#license)

## Quick start

```bash
git clone https://github.com/vallematteo/flood_cascades.git && cd flood_cascades
julia +1.10.4 --project=. -e 'using Pkg; Pkg.instantiate()'   # Julia packages, about 2 minutes
uv sync && source .venv/bin/activate                          # Python packages, about 10 seconds
bash reproduce/reproduce.sh                                   # every figure, table and number, about 7 minutes
```

To run the model itself, the [demo](REPRODUCE.md#demo) runs 20 simulations and checks them
against `data/` in under a minute.

## Overview

This repository contains an agent-based model of mortgage default calibrated to the Limburg area in the Netherlands. The model is written in Julia with Agents.jl; the analysis that turns simulation
output into the paper's figures, tables and numbers is written in Python.

### Choose what to reproduce

The three levels are entry points into the same workflow, depending on how much you want
to rerun. They are not steps to follow in alphabetical order. The full workflow is:

```text
C. Run simulations → raw simulation output
                              ↓
B. Summarise raw output → reduced data (the quantities stored in data/)
                              ↓
A. Generate paper results → figures, result tables and quoted numbers
```

**Start with A to reproduce the paper's results.** The repository already includes the
reduced data, so you can run A without running B or C. Raw simulation output is not included;
to rebuild the reduced data, run C then B for the relevant experiment.

| Entry point and goal | Required inputs | What you generate | How to run |
|---|---|---|---|
| **A. Regenerate paper results** | Published reduced data in `data/`, plus the supplied model inputs used by the analysis scripts | Data-based figures, result tables as LaTeX tabulars, and `numbers.csv` in `output/`, covering the main text and Supplementary Information (SI) | `bash reproduce/reproduce.sh`; about 7 minutes on a laptop |
| **B. Rebuild the reduced data** | Raw simulation output from C and the experiment's run list/design | Summaries such as per-run foreclosure rates, pooled statistics and cascade sizes, saved in `output/<experiment>/reduced/` for comparison with `data/<experiment>/` | Follow the reduction commands in the relevant [`experiments/`](experiments) README; about 1 to 30 minutes per experiment on a cluster node |
| **C. Rerun the model experiments** | Model code, prepared inputs, parameter designs and random seeds | Raw simulation records for each run, which B then summarises | Follow the design and simulation commands in the relevant [`experiments/`](experiments) README; laptop for the demo, foreclosure timing and liquid-funds scale experiments; SLURM cluster used for the full calibration, sensitivity analysis and cascades experiments |

Here, **raw output** means the simulation records before analysis; **reduced data** means
the selected and aggregated quantities used by the paper's analysis scripts. Reruns write
to `output/`, leaving the published reference files in `data/` unchanged.

A generates the scripted results, not the manuscript PDF, the model schematic (Fig. 1),
or the descriptive SI tables. See [REPRODUCE.md](REPRODUCE.md) for the exact coverage,
known numerical/rendering differences and computing costs for B and C. To try C and B on
a small example first, use the [20-simulation demo](REPRODUCE.md#demo).

## Repository contents

| Path | Content |
|---|---|
| [`model/`](model) | the agent-based model (`model.jl`, `agents.jl`, `utils.jl`) and `parameters.csv`, the fixed, baseline and calibrated parameter values that every runner and script reads |
| [`preparation/`](preparation) | model inputs and their preparation: flood maps, flood-severity series, initialization (CBS data and baseline parameters) and the calibration targets (NHG, BKR) ([preparation/README.md](preparation/README.md)) |
| [`experiments/`](experiments) | one folder per simulation experiment (calibration, global sensitivity analysis, cascades, foreclosure timing, scale of liquid funds), each with its parameter design, runner, SLURM script where it ran on a cluster, a script that summarises the raw output into `data/`, and README |
| [`data/`](data) | summarised simulation output (raw runs aggregated to the quantities the paper uses), the only data the figure, table and number scripts read ([data/README.md](data/README.md)) |
| [`reproduce/`](reproduce) | the scripts that turn `data/` into the paper (level A), writing to `output/`: |
| `reproduce/figure_*.py` | one script per figure, named by its number in the paper (`figure_2.py`) or the SI (`figure_SI1.py`); Fig. 1 is a schematic (drawn in TikZ, no data) and has no script |
| [`reproduce/tables.py`](reproduce/tables.py) | writes Table 1 and SI Tables 5, 6, 8, 9 and 10 as LaTeX tabulars |
| [`reproduce/paper_numbers.py`](reproduce/paper_numbers.py) | recomputes every number quoted in the paper and the SI and lists them in `output/numbers.csv` |
| [`reproduce/utils.py`](reproduce/utils.py) | code shared by these scripts (paths, constants, data loaders, statistics, plot style) |
| [`reproduce/reproduce.sh`](reproduce/reproduce.sh) | runs every figure script, `tables.py` and `paper_numbers.py` |
| [`REPRODUCE.md`](REPRODUCE.md) | for each figure, table and number block: the script, the data, the run time, and how to regenerate the data |

## System requirements

### Hardware

**No cluster is needed for level A:** all scripted paper results can be generated from
the supplied data on a laptop. For level C, the table below distinguishes the experiments
run locally from the larger ensembles run on a cluster for the paper.

| Task | Number of simulations | Hardware used | Measured elapsed time |
|---|---|---|---|
| A: generate paper results | none | laptop | about 7 min |
| Demo: simulate and reduce the calibrated no-flood case | 20 | laptop, 4 workers | about 40 s |
| C: foreclosure timing | 1,050 | laptop, 10 workers | 13–15 min |
| C: scale of initial liquid funds | 505 | laptop, 8 workers | 7 min |
| C + B: calibration, including reduction | 10,010 | cluster, one node with 192 workers | 11 min |
| C: global sensitivity analysis | 166,400 | cluster, 5 array tasks, each using one node with 192 workers | 28–31 min per task |
| C: foreclosure cascades | 65,520 | cluster, one node with 192 workers | 75 min |

Laptop timings were measured on an Apple M3 Pro (11 cores, 18 GB RAM). Cluster timings
are execution times on Snellius (SURF, the Netherlands), excluding queue time; the five
sensitivity-analysis tasks may run concurrently or separately depending on availability.
Each node has 2 AMD EPYC 9654 processors, 192 cores and 1.4 TiB of memory; the simulation
jobs requested 336 GB per node.

**The cluster is used to run many independent simulations in parallel.** The Julia runners
can also run locally with fewer workers; the experiment READMEs provide local commands
and SLURM submission commands where applicable. One simulation takes about 8–9 s on one
core and needs up to about 1.2 GB of memory. Budget at least 1.5 GB per worker plus a few
GB for the main process. A laptop can therefore run subsets of any experiment, while full
ensembles take longer with fewer workers.

For level B, reduce the output on a machine with access to the raw files. The full
sensitivity-analysis reduction took 28 min on 192 cluster cores; cascade reduction and
tree reconstruction took 1 and 2 min on 24 and 72 cores respectively. Local reduction
commands are also provided. Full reruns require storage as well as compute: estimated raw
output is about 0.8 TB for sensitivity analysis, 0.1 TB for calibration and 1.4 GB for
cascades. See [REPRODUCE.md](REPRODUCE.md#computing-resources) for detailed resource costs
and the [experiment READMEs](experiments) for commands.

### Software

Tested on:

- macOS 27.2 (Apple M3 Pro)
- Snellius: Red Hat Enterprise Linux 9.8 (x86-64)

Dependencies (exact versions are pinned in `Manifest.toml` for Julia and in `pyproject.toml` / `uv.lock` for Python):

- Julia 1.10.4, with Agents.jl 7.0.1 and the packages in `Project.toml`. Use exactly this
  version: other Julia versions draw different random numbers, so the simulations differ
  slightly from those in the paper (with [juliaup](https://github.com/JuliaLang/juliaup):
  `juliaup add 1.10.4`, then `julia +1.10.4`)
- Python 3.11, managed with [uv](https://docs.astral.sh/uv/): NumPy 1.26.4, pandas 2.3.3, SciPy 1.11.4, Matplotlib 3.10.8, SALib 1.4.7, Pillow 10.2.0, PyArrow 23.0.0

No non-standard hardware is required.

## Installation guide

```bash
git clone https://github.com/vallematteo/flood_cascades.git
cd flood_cascades

# Julia environment (Julia 1.10.4)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Python environment (uv installs Python 3.11 and the pinned packages into .venv)
uv sync
source .venv/bin/activate
```

Every `python` command in this README, in REPRODUCE.md and in the scripts assumes this
environment is active (`source .venv/bin/activate` in each new shell).

Install time, measured on an Apple M3 Pro with a fast connection and empty caches: about 10
seconds for the Python environment and about 2 minutes for the Julia environment (download
and precompilation of the packages).

## Demo

The 20 simulations of the calibrated parametrisation without floods, summarised the same way as `data/`,
reproduce the annual foreclosure rate (0.42%) and the share of households in financial distress
(0.51%) of the main text in about 40 seconds on 4 cores. Commands, expected output and run time:
[REPRODUCE.md](REPRODUCE.md#demo).

## Reproducing the paper

```bash
bash reproduce/reproduce.sh   # every figure, table and number from data/ (level A), about 7 minutes on a laptop
```

[REPRODUCE.md](REPRODUCE.md) lists each figure, table and number block with its script, data,
run time, and how to regenerate its data from the simulations.

## License

Code: MIT License (see [LICENSE](LICENSE)). Data in `data/`: Creative Commons Attribution 4.0 (CC BY 4.0). Third-party data
in `preparation/` keeps the licence of its source (see [preparation/README.md](preparation/README.md)).
