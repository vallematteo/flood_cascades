"""Reduce the output of the foreclosure-timing runs to the variables the paper uses.

The runner (run.jl) writes two tables per design:
    <design>_runs.csv           one row per run (parameters and run totals)
    <design>_foreclosures.csv   one row per foreclosure, with the run parameters repeated
This script keeps the run parameters only in the runs table (joined by run_id) and, per
foreclosure, only the variables used by Fig. 5, SI Figs 11 and 12 and the quoted numbers.

Designs: gap (beta1 - beta2 from 0.45 to 0.7) and beta3 (beta3 from 0.5 to 2).

Run:    python experiments/foreclosure_timing/reduce_foreclosure_timing.py [--runner-dir output/foreclosure_timing/raw]
Writes: data/foreclosure_timing/{gap,beta3}_runs.csv
        data/foreclosure_timing/{gap,beta3}_foreclosures.csv.gz
"""
import argparse
import os

import pandas as pd

GZIP = {"method": "gzip", "mtime": 0}   # fixed timestamp: identical content gives identical files
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "foreclosure_timing")

DESIGNS = ["gap", "beta3"]

RUN_COLUMNS = ["run_id", "setting", "seed", "beta1", "beta2", "beta3", "lambda", "delta",
               "eta", "m", "varepsilon", "n_trig_fin", "n_trig_strat", "n_fcl_fin",
               "n_fcl_strat", "n_fcl_tot", "mean_active", "agents_end", "warmup"]

FORECLOSURE_COLUMNS = [
    "run_id",
    "step",            # model month of the foreclosure
    "t",               # age of the mortgage at foreclosure (months)
    "id",              # household
    "default_reason",  # 1 = financial, -1 = strategic
    "ltv",             # loan-to-value at foreclosure
    "n_dis",           # distress months before and including the foreclosure
    "n_dis_fin",       # of which financial triggers
    "n_dis_strat",     # of which strategic triggers
    "t_dis_first",     # mortgage age at the first distress month (-1 if none)
]


def reduce_design(runner_dir, design):
    runs = pd.read_csv(os.path.join(runner_dir, f"{design}_runs.csv"), float_precision="round_trip")
    fcl = pd.read_csv(os.path.join(runner_dir, f"{design}_foreclosures.csv"), float_precision="round_trip")
    assert fcl.warmup.nunique() == 1, "one warm-up per design"
    runs = runs[runs.success].assign(warmup=fcl.warmup.iloc[0])
    return runs[RUN_COLUMNS], fcl[FORECLOSURE_COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner-dir", default=os.path.join(ROOT, "output", "foreclosure_timing", "raw"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in DESIGNS:
        runs, fcl = reduce_design(a.runner_dir, name)
        runs.to_csv(os.path.join(OUT, f"{name}_runs.csv"), index=False)
        fcl.to_csv(os.path.join(OUT, f"{name}_foreclosures.csv.gz"), index=False, compression=GZIP)
        print(f"{name}: {len(runs)} runs, {len(fcl):,} foreclosures")


if __name__ == "__main__":
    main()
