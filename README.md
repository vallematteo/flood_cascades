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
- [Citation](#citation)

## Quick start

```bash
git clone https://github.com/vallematteo/flood_cascades.git && cd flood_cascades
julia +1.10.4 --project=. -e 'using Pkg; Pkg.instantiate()'   # Julia packages, about 2 minutes
uv sync && source .venv/bin/activate                          # Python packages, about 10 seconds
bash reproduce.sh                                             # every figure, table and number, about 7 minutes
```

To run the model itself, the [demo](REPRODUCE.md#demo) runs 20 simulations and checks them
against `data/` in under a minute.

## Overview

This repository contains an agent-based model of mortgage default calibrated to the Limburg area in the Netherlands. The model is written in Julia with Agents.jl; the analysis that turns simulation
output into the paper's figures, tables and numbers is written in Python.

The package can be used at three levels:

| Level | What it reproduces | Where it runs | Time |
|---|---|---|---|
| A. Paper from data | every figure, table and number in the paper and SI, from the summarised simulation output in `data/` | a laptop | about 7 minutes |
| B. Data from raw output | the summarised output in `data/`, from the raw simulation output | wherever the raw output is | 1 to 30 minutes per experiment on a cluster node (see [REPRODUCE.md](REPRODUCE.md)) |
| C. Simulations | the raw output, from the model | laptop (foreclosure timing, demo runs); SLURM cluster (calibration, sensitivity analysis, cascades) | see [REPRODUCE.md](REPRODUCE.md) |

## Repository contents

| Path | Content |
|---|---|
| [`model/`](model) | the agent-based model (`model.jl`, `agents.jl`, `utils.jl`) and `parameters.csv`, the fixed, baseline and calibrated parameter values that every runner and script reads |
| [`preparation/`](preparation) | model inputs and their preparation: flood maps, flood-severity series, initialization (CBS data and baseline parameters) and the calibration targets (NHG, BKR) ([preparation/README.md](preparation/README.md)) |
| [`experiments/`](experiments) | one folder per simulation experiment (calibration, global sensitivity analysis, cascades, foreclosure timing, scale of liquid funds), each with its parameter design, runner, SLURM script where it ran on a cluster, a script that summarises the raw output into `data/`, and README |
| [`data/`](data) | summarised simulation output (raw runs aggregated to the quantities the paper uses), the only data the figure, table and number scripts read ([data/README.md](data/README.md)) |
| `figure_*.py` | one script per figure, named by its number in the paper (`figure_2.py`) or the SI (`figure_SI1.py`); Fig. 1 is a schematic (drawn in TikZ, no data) and has no script |
| `tables.py` | writes Table 1 and SI Tables 5, 6, 8, 9 and 10 as LaTeX tabulars |
| `paper_numbers.py` | recomputes every number quoted in the paper and the SI and lists them in `output/numbers.csv` |
| `utils.py` | code shared by several scripts (paths, constants, data loaders, statistics, plot style) |
| `reproduce.sh` | runs every figure script, `tables.py` and `paper_numbers.py` (level A) |
| [`REPRODUCE.md`](REPRODUCE.md) | for each figure, table and number block: the script, the data, the run time, and how to regenerate the data |

## System requirements

### Hardware

Level A and the demo need only a standard computer. The timings in this README and in
REPRODUCE.md were measured on an Apple M3 Pro (11 cores, 18 GB RAM).

The calibration, the sensitivity analysis and the cascade experiment ran on the Snellius
national supercomputer (SURF, the Netherlands), one node at a time: 2 AMD EPYC 9654 processors,
192 cores and 1.4 TiB of memory per node. One simulation needs about 1.2 GB of memory and 8 to 9
s on one core, so any machine with at least 1.5 GB per core can run them; see
[REPRODUCE.md](REPRODUCE.md#computing-resources).

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
bash reproduce.sh        # every figure, table and number from data/ (level A), about 7 minutes on a laptop
```

[REPRODUCE.md](REPRODUCE.md) lists each figure, table and number block with its script, data,
run time, and how to regenerate its data from the simulations.

## License

Code: MIT License (see [LICENSE](LICENSE)). Data in `data/`: Creative Commons Attribution 4.0 (CC BY 4.0). Third-party data
in `preparation/` keeps the licence of its source (see [preparation/README.md](preparation/README.md)).


