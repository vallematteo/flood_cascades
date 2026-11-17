"""Merge the per-run tables of the cascade runs and pair every run with its delta = 0 run.

Every run is paired with the delta = 0 run of the same (setting, lambda, kappa, beta2,
vartheta, seed), and every flooded run with the S0 run of the same (kappa, beta2, vartheta,
delta, seed):

    R_net   = 1 - rate(delta = 0) / rate(delta)           net reproduction number
    C       = ln rate(delta) - ln rate(delta = 0)         cascade term, = -ln(1 - R_net)
    R_gross = attributed foreclosures / foreclosures
    flood   = ln rate(S, lambda, 0) - ln rate(S0, 0)      change of the background rate
    C_S0    = C(S0, delta)                                cascade term without floods
    I       = C(S, lambda, delta) - C(S0, delta)          flood x cascade interaction
    total   = ln rate(S, lambda, delta) - ln rate(S0, 0) = flood + C_S0 + I

Rates are annual foreclosures per 100 active mortgages over the post-warm-up window.

Run:    python experiments/cascades_rnet/reduce_runs.py --raw-dir <folder with runs_chunk*.csv>
Writes: data/cascades_rnet/runs_all.csv.gz     all successful runs, one row each
        data/cascades_rnet/paired_seed.csv.gz  per (cell, seed): rates, R_net, C, R_gross, flood, C_S0, I
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

GZIP = {"method": "gzip", "mtime": 0}   # fixed timestamp: identical content gives identical files
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "cascades_rnet")
CELL = ["setting", "lambda", "kappa", "beta2", "vartheta", "delta"]
BASE = ["setting", "lambda", "kappa", "beta2", "vartheta"]      # cell without delta


def read_runs(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "runs_chunk*.csv")))
    if not files:
        raise SystemExit(f"no runs_chunk*.csv in {raw_dir}")
    runs = pd.concat((pd.read_csv(f, float_precision="round_trip") for f in files), ignore_index=True)
    failed = runs[~runs["success"].astype(bool)]
    if len(failed):
        print(f"WARNING: {len(failed)} failed runs dropped: {failed['run_id'].tolist()[:20]}")
    runs = runs[runs["success"].astype(bool)].copy()
    assert not runs.duplicated("run_id").any(), "duplicate run_id across chunks"
    assert not runs.duplicated(CELL + ["seed"]).any(), "duplicate (cell, seed)"
    return runs


def pair(runs):
    r = runs.copy()
    r["ln_rate"] = np.log(r["rate_tot"].where(r["rate_tot"] > 0))
    r["R_gross"] = r["n_attr_tot"] / r["n_fcl_tot"].where(r["n_fcl_tot"] > 0)
    r["strat_share"] = r["n_fcl_strat"] / r["n_fcl_tot"].where(r["n_fcl_tot"] > 0)

    # with the delta = 0 run of the same cell and seed
    zero = r[r["delta"] == 0][BASE + ["seed", "rate_tot", "ln_rate", "rate_fin", "rate_strat"]]
    zero = zero.rename(columns={"rate_tot": "rate0", "ln_rate": "ln_rate0",
                                "rate_fin": "rate_fin0", "rate_strat": "rate_strat0"})
    p = r.merge(zero, on=BASE + ["seed"], how="left", validate="many_to_one")
    p["R_net"] = 1 - p["rate0"] / p["rate_tot"]
    p["C"] = p["ln_rate"] - p["ln_rate0"]
    p["d_rate_fin"] = p["rate_fin"] - p["rate_fin0"]
    p["d_rate_strat"] = p["rate_strat"] - p["rate_strat0"]

    # with the S0 run of the same (kappa, beta2, vartheta, delta, seed)
    s0_key = ["kappa", "beta2", "vartheta", "delta", "seed"]
    s0 = p[p["setting"] == 0][s0_key + ["ln_rate", "ln_rate0", "C"]]
    s0 = s0.rename(columns={"ln_rate": "ln_rate_S0", "ln_rate0": "ln_rate0_S0", "C": "C_S0"})
    p = p.merge(s0, on=s0_key, how="left", validate="many_to_one")
    p["flood"] = p["ln_rate0"] - p["ln_rate0_S0"]
    p["I"] = p["C"] - p["C_S0"]
    p["total"] = p["ln_rate"] - p["ln_rate0_S0"]
    gap = (p["total"] - (p["flood"] + p["C_S0"] + p["I"])).abs().max()
    assert not (gap > 1e-9), f"decomposition does not add up ({gap})"

    keep = CELL + ["seed", "seed_index", "rate_tot", "rate0", "rate_fin", "rate_strat",
                   "d_rate_fin", "d_rate_strat", "R_net", "C", "R_gross", "strat_share",
                   "flood", "C_S0", "I", "total", "mean_active", "mean_nbox_fcl"]
    return p[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    runs = read_runs(a.raw_dir)
    runs.to_csv(os.path.join(OUT, "runs_all.csv.gz"), index=False, compression=GZIP)
    paired = pair(runs)
    paired.to_csv(os.path.join(OUT, "paired_seed.csv.gz"), index=False, compression=GZIP)
    print(f"{len(runs):,} runs -> runs_all.csv.gz, paired_seed.csv.gz in {OUT}")


if __name__ == "__main__":
    main()
