"""Parameter combinations of the calibration (rejection ABC) at the no-flood scenario.

Prior: uniform on the sensitivity-analysis ranges of the nine parameters that act without
floods when delta = 0 (kappa, beta1, beta2, beta3, eta, vartheta, m, varepsilon, exp_lb),
restricted to beta1 - beta2 >= 1/2. The combinations are the points of a scrambled Sobol'
sequence of 2^13 points (scrambling seed 20261104) that fall inside this region: 1,001
combinations. Each runs with the ten seeds of the sensitivity analysis at S0, with delta = 0
(no spillover) and lambda = 0.25 (no effect without floods): 10,010 runs.

Run:    uv run python experiments/abc_calibration/design.py
Writes: data/abc_calibration/design.csv       sample_id, raw_index (position in the Sobol'
                                              sequence) and the nine parameters
        output/abc_calibration/run_list.csv   one row per run, read by experiments/global_sa/run.jl
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BOUNDS = os.path.join(ROOT, "data", "global_sa", "design", "problem_definition.csv")
DESIGN = os.path.join(ROOT, "data", "abc_calibration", "design.csv")
RUN_LIST = os.path.join(ROOT, "output", "abc_calibration", "run_list.csv")

PARAMS = ["kappa", "beta1", "beta2", "beta3", "eta", "vartheta", "m", "varepsilon", "exp_lb"]
MIN_GAP = 0.5            # beta1 - beta2 >= 1/2
SOBOL_SEED = 20261104
M = 13                   # 2^13 points of the Sobol' sequence
SEEDS = np.random.default_rng(56).integers(0, 2**31, size=10).tolist()   # as the sensitivity analysis
FIXED = {"setting": 0, "lambda": 0.25, "delta": 0.0}


def main():
    bounds = pd.read_csv(BOUNDS).set_index("parameter")
    lo = bounds.loc[PARAMS, "lower_bound"].to_numpy()
    hi = bounds.loc[PARAMS, "upper_bound"].to_numpy()
    u = qmc.Sobol(d=len(PARAMS), scramble=True, seed=SOBOL_SEED).random_base2(m=M)
    x = pd.DataFrame(lo + u * (hi - lo), columns=PARAMS)
    x.insert(0, "raw_index", np.arange(len(x)))
    x = x[x.beta1 - x.beta2 >= MIN_GAP].reset_index(drop=True)
    x.insert(0, "sample_id", np.arange(len(x)))
    os.makedirs(os.path.dirname(DESIGN), exist_ok=True)
    x.to_csv(DESIGN, index=False)

    runs = x.loc[x.index.repeat(len(SEEDS))].reset_index(drop=True)
    runs["seed_index"] = np.tile(np.arange(len(SEEDS)), len(x))
    runs["seed"] = np.tile(SEEDS, len(x))
    runs["run_id"] = len(SEEDS) * runs.sample_id + runs.seed_index
    for c, v in FIXED.items():
        runs[c] = v
    runs = runs[["run_id", "sample_id", "setting", "seed_index", "seed", *PARAMS, "lambda", "delta"]]
    os.makedirs(os.path.dirname(RUN_LIST), exist_ok=True)
    runs.to_csv(RUN_LIST, index=False)
    print(f"{2 ** M} Sobol' points, {len(x)} with beta1 - beta2 >= {MIN_GAP}; {len(runs):,} runs -> {RUN_LIST}")


if __name__ == "__main__":
    main()
