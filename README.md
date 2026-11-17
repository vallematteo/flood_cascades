# Flood cascades

Code and data for the paper *Neighbourhood flood damage amplifies mortgage default cascades*.

> **Status: work in progress.** Sections marked `TODO` are filled in as the package is built.

## Contents

- [Overview](#overview)
- [Repository contents](#repository-contents)
- [System requirements](#system-requirements)
- [Installation guide](#installation-guide)
- [Demo](#demo)
- [Reproducing the paper](#reproducing-the-paper)
- [License](#license)
- [Citation](#citation)

## Overview

An agent-based model of mortgage holders in Limburg (the Netherlands) exposed to floods. Each
household holds a mortgage, receives income, and may default through a financial channel
(it cannot pay) or a strategic channel (it chooses not to pay after its home or its
neighbourhood loses value). Foreclosures can spill over to neighbours. The model is written in
Julia with Agents.jl; the analysis that turns simulation output into the paper's figures,
tables and numbers is written in Python.

The package can be used at three levels:

| Level | What it reproduces | Where it runs | Time |
|---|---|---|---|
| A. Paper from data | every figure, table and number in the paper and SI, from the reduced data in `data/` | a laptop | TODO minutes |
| B. Data from raw output | the reduced data in `data/`, from raw simulation output | wherever the raw output is | TODO |
| C. Simulations | the raw output, from the model | laptop (foreclosure timing, demo runs); SLURM cluster (calibration, sensitivity analysis, cascades) | see [REPRODUCE.md](REPRODUCE.md) |

## Repository contents

| Path | Content |
|---|---|
| [`model/`](model) | the agent-based model (`model.jl`, `agents.jl`, `utils.jl`) |
| [`preparation/`](preparation) | model inputs and their preparation: flood maps, flood-severity series, initialization (CBS data and baseline parameters) and the calibration targets (NHG, BKR) ([preparation/README.md](preparation/README.md)) |
| `experiments/` | TODO: one folder per experiment, with its parameter design, simulation runner, SLURM script and reduction script |
| `data/` | TODO: reduced simulation output, the only data the figure and number scripts read |
| `figure_*.py` | TODO: one script per figure, named by its number in the paper (`figure_2.py`) or SI (`figure_S1.py`) |
| `tables.py` | TODO: writes every results table |
| `paper_numbers.py` | TODO: recomputes every number quoted in the paper and checks it against the printed value |
| `utils.py` | TODO: code shared by several scripts (paths, data loaders, statistics, plot style) |
| [`REPRODUCE.md`](REPRODUCE.md) | for each figure, table and number block: the script, the data, the run time, and how to regenerate the data |

## System requirements

### Hardware

Level A and the demo need only a standard computer. The timings in this README and in
REPRODUCE.md were measured on an Apple M3 Pro (11 cores, 18 GB RAM).

The full simulation experiments were run on a SLURM cluster (TODO: node type, cores and memory
per node).

### Software

Tested on:

- macOS 27.2 (Apple M3 Pro)
- TODO: cluster operating system

Dependencies (exact versions are pinned in `Manifest.toml` for Julia and in `pyproject.toml` / `uv.lock` for Python):

- Julia 1.10.4, with Agents.jl 7.0.1 and the packages in `Project.toml`
- Python 3.11, managed with [uv](https://docs.astral.sh/uv/): NumPy 1.26.4, pandas 2.3.3, SciPy 1.11.4, Matplotlib 3.10.8, SALib 1.4.7, Pillow 10.2.0, PyArrow 23.0.0

No non-standard hardware is required.

## Installation guide

TODO: commands and typical install time.

```bash
git clone https://github.com/vallematteo/flood_cascades.git
cd flood_cascades

# Julia environment (Julia 1.10.4)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Python environment (uv installs Python 3.11 and the pinned packages into .venv)
uv sync
```

## Demo

TODO: one short simulation, its command, expected output and run time.

## Reproducing the paper

TODO. In short:

```bash
bash reproduce.sh        # every figure, table and number from data/ (level A)
```

Run every Python script through uv, for example `uv run python figure_2.py`, so it uses the
pinned environment.

[REPRODUCE.md](REPRODUCE.md) lists each figure, table and number block with its script, data,
run time, and how to regenerate its data from the simulations.

## License

Code: MIT License (see [LICENSE](LICENSE)). Data in `data/`: TODO (CC-BY 4.0). Third-party data
in `preparation/` keeps the licence of its source (see [preparation/README.md](preparation/README.md)).

## Citation

TODO: paper reference and archive DOI. See [CITATION.cff](CITATION.cff).
