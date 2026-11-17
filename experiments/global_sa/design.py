"""Saltelli design of the global sensitivity analysis.

Eleven parameters are sampled (kappa, beta1, beta2, beta3, lambda, delta, eta, vartheta, m,
varepsilon, exp_lb) with N = 256 base samples and first-order and total-order indices only, so
the design has N (D + 2) = 256 x 13 = 3,328 parameter vectors. Every vector satisfies
beta1 > beta2 (the A and B samples are filtered jointly, and every AB_j block is rebuilt from
the filtered A and B). The retained subset of the Sobol' sequence has no low-discrepancy
guarantee, so five candidate designs (sampling seeds 56 to 60) are generated and the one with
the lowest centered L2-discrepancy is kept.

Each vector runs in the five flood scenarios with ten seeds, np.random.default_rng(56)
.integers(0, 2**31, 10): 3,328 x 5 x 10 = 166,400 runs.

Run:    uv run python experiments/global_sa/design.py
Writes: data/global_sa/design/problem_definition.csv   parameter names and sampling ranges
        data/global_sa/design/saltelli_matrix.csv      the 3,328 vectors, by saltelli_sample_id
        data/global_sa/design/design_config.csv        candidate seeds, discrepancies, the chosen seed
        output/global_sa/run_list.csv                  one row per run (run_id, scenario, seed, vector), read by run.jl
"""
import math
import os

import numpy as np
import pandas as pd
from SALib.sample import sobol

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DESIGN_DIR = os.path.join(ROOT, "data", "global_sa", "design")
RUN_LIST = os.path.join(ROOT, "output", "global_sa", "run_list.csv")

N_SAMPLES = 256
N_CANDIDATES = 5
BASE_SEED = 56          # first candidate sampling seed, and the seed of the replication seeds
N_SEEDS = 10
SETTINGS = [0, 1, 2, 3, 4]

# Baseline values that centre the ranges of m, varepsilon and exp_lb (initialization).
M_BASE, VAREPSILON_BASE, EXP_LB_BASE = 6.2, 0.00198, 1087

PROBLEM = {
    "num_vars": 11,
    "names": ["kappa", "beta1", "beta2", "beta3", "lambda", "delta",
              "eta", "vartheta", "m", "varepsilon", "exp_lb"],
    "bounds": [
        [1, 25],                                        # kappa: neighbourhood radius
        [0.01, 1],                                      # beta1
        [0.01, 1],                                      # beta2
        [0.25, 3.0],                                    # beta3: half-saturation debt pressure
        [0.0, 0.5],                                     # lambda: share of house value lost at D = 1
        [0.05, 0.5],                                    # delta: discount per nearby foreclosure
        [0.03, 0.13],                                   # eta
        [0.03, 0.13],                                   # vartheta
        [M_BASE * 0.7, M_BASE * 1.3],                   # m
        [VAREPSILON_BASE * 0.7, VAREPSILON_BASE * 1.3], # varepsilon
        [EXP_LB_BASE * 0.7, EXP_LB_BASE * 1.3],         # exp_lb
    ],
}


def sobol_sample_constrained(problem, n, seed):
    """Saltelli matrix [A ; B ; AB_0 ; ... ; AB_{D-1}] with beta1 > beta2 in every row.

    SALib's sobol.sample returns the matrix interleaved per base sample,
    [A_i, AB_0_i, ..., AB_{D-1}_i, B_i]; A and B are read as the first and last row of each
    base sample, filtered jointly, and every AB_j is rebuilt as A with column j taken from B.
    """
    i_b1, i_b2 = problem["names"].index("beta1"), problem["names"].index("beta2")
    D = problem["num_vars"]
    batch_n = 2 ** math.ceil(math.log2(n * 20))     # next power of 2 >= 20 n
    full = sobol.sample(problem, batch_n, calc_second_order=False, seed=seed).reshape(batch_n, D + 2, D)
    A_all, B_all = full[:, 0, :], full[:, -1, :]
    a1, a2, b1, b2 = A_all[:, i_b1], A_all[:, i_b2], B_all[:, i_b1], B_all[:, i_b2]
    valid = np.where((a1 > a2) & (b1 > b2) & (b1 > a2) & (a1 > b2))[0]
    assert len(valid) >= n, f"only {len(valid)} valid (A, B) pairs, need {n}"
    A, B = A_all[valid[:n]], B_all[valid[:n]]

    result = np.zeros((n * (D + 2), D))
    result[0:n] = A
    result[n:2 * n] = B
    for j in range(D):
        AB_j = A.copy()
        AB_j[:, j] = B[:, j]
        result[(j + 2) * n:(j + 3) * n] = AB_j
    assert (result[:, i_b1] > result[:, i_b2]).all(), "beta1 > beta2 violated"
    return result


def centered_l2_discrepancy(X, bounds):
    """Centered L2-discrepancy (Hickernell 1998) after mapping each column to [0, 1]; lower is more uniform."""
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    Z = (np.asarray(X, dtype=float) - lo) / (hi - lo)
    n, d = Z.shape
    dev = np.abs(Z - 0.5)
    term1 = (13.0 / 12.0) ** d
    term2 = (2.0 / n) * np.prod(1.0 + 0.5 * dev - 0.5 * dev ** 2, axis=1).sum()
    total = 0.0
    for a in range(0, n, 256):                       # pairwise term in chunks of 256 rows
        Zi = Z[a:a + 256][:, None, :]
        di = dev[a:a + 256][:, None, :]
        prod = 1.0 + 0.5 * di + 0.5 * dev[None, :, :] - 0.5 * np.abs(Zi - Z[None, :, :])
        total += np.prod(prod, axis=2).sum()
    return float(np.sqrt(max(term1 - term2 + total / (n * n), 0.0)))


def main():
    candidates = []
    for seed in range(BASE_SEED, BASE_SEED + N_CANDIDATES):
        values = sobol_sample_constrained(PROBLEM, N_SAMPLES, seed)
        candidates.append((seed, centered_l2_discrepancy(values, PROBLEM["bounds"]), values))
        print(f"candidate seed {seed}: CD2 = {candidates[-1][1]:.6f}")
    chosen_seed, chosen_cd2, values = min(candidates, key=lambda c: c[1])
    print(f"chosen: seed {chosen_seed}, CD2 = {chosen_cd2:.6f}")

    names = PROBLEM["names"]
    os.makedirs(DESIGN_DIR, exist_ok=True)
    pd.DataFrame({"parameter": names,
                  "lower_bound": [b[0] for b in PROBLEM["bounds"]],
                  "upper_bound": [b[1] for b in PROBLEM["bounds"]]}).to_csv(
        os.path.join(DESIGN_DIR, "problem_definition.csv"), index=False)
    matrix = pd.DataFrame(values, columns=names)
    matrix.insert(0, "saltelli_sample_id", range(len(values)))
    matrix.to_csv(os.path.join(DESIGN_DIR, "saltelli_matrix.csv"), index=False)
    pd.DataFrame([{"n_samples": N_SAMPLES, "n_vectors": len(values), "n_candidates": N_CANDIDATES,
                   "candidate_cd2": ";".join(f"{s}:{cd:.6f}" for s, cd, _ in candidates),
                   "chosen_sample_seed": chosen_seed, "chosen_cd2": chosen_cd2}]).to_csv(
        os.path.join(DESIGN_DIR, "design_config.csv"), index=False)

    # one row per run: scenario, then vector, then seed
    seeds = np.random.default_rng(BASE_SEED).integers(0, 2**31, size=N_SEEDS).tolist()
    rows = []
    for setting in SETTINGS:
        for idx, v in enumerate(values):
            for seed_index, seed in enumerate(seeds):
                rows.append({"run_id": len(rows), "saltelli_sample_id": idx, "setting": setting,
                             "seed_index": seed_index, "seed": seed, **dict(zip(names, v))})
    os.makedirs(os.path.dirname(RUN_LIST), exist_ok=True)
    pd.DataFrame(rows).to_csv(RUN_LIST, index=False)
    print(f"{len(values)} vectors x {len(SETTINGS)} scenarios x {N_SEEDS} seeds = {len(rows):,} runs -> {RUN_LIST}")


if __name__ == "__main__":
    main()
