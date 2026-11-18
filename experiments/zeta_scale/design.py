"""Run list of the test of the scale of initial liquid funds (zeta_scale).

Each household draws the ratio of its liquid funds to its loan from a fitted Burr distribution,
multiplied by zeta_scale (0.015 in every other simulation). The test sweeps zeta_scale over
101 values from 0.005 to 0.03 in steps of 0.00025, with the first five seeds of the sensitivity
analysis: 505 runs. All runs are at the no-flood scenario with delta = 0 (no spillover), at the
calibrated parametrisation and the baseline values of the other parameters (beta1 = 1,
beta2 = 0.5, beta3 = 1, kappa = 10, vartheta = 0.05, exp_lb = 1087). lambda has no effect
without floods.

Run:    python experiments/zeta_scale/design.py
Writes: output/zeta_scale/run_list.csv   one row per run, read by experiments/global_sa/run.jl
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "model"))
from parameters import model_parameters  # noqa: E402
RUN_LIST = os.path.join(ROOT, "output", "zeta_scale", "run_list.csv")

ZETA_SCALES = np.round(np.arange(0.005, 0.030 + 1e-12, 0.00025), 6)   # 101 values
SEEDS = np.random.default_rng(56).integers(0, 2**31, size=10)[:5].tolist()   # first five seeds of the sensitivity analysis
_P = model_parameters("baseline", "calibrated")       # model/parameters.csv
FIXED = {"setting": 0,
         **{p: _P[p] for p in ("kappa", "beta1", "beta2", "beta3", "eta", "vartheta", "m", "varepsilon", "exp_lb")},
         "lambda": 0.5, "delta": _P["delta"]}


def main():
    rows = [{"run_id": len(SEEDS) * i + k, "sample_id": i, "seed_index": k, "seed": seed,
             "zeta_scale": z, **FIXED}
            for i, z in enumerate(ZETA_SCALES) for k, seed in enumerate(SEEDS)]
    runs = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(RUN_LIST), exist_ok=True)
    runs.to_csv(RUN_LIST, index=False)
    print(f"{len(ZETA_SCALES)} values of zeta_scale x {len(SEEDS)} seeds = {len(runs)} runs -> {RUN_LIST}")


if __name__ == "__main__":
    main()
