"""Foreclosures per map cell, for the foreclosure-rate maps of SI Fig. 2.

Reads the raw foreclosure records written by the cascade runs (fcl_chunk*.csv, one row per
foreclosure with its map cell x, y) and keeps, for the runs used by the maps, the number of
post-warm-up foreclosures in each cell of each run. The maps are smoothed histograms of
foreclosure locations, and smoothing is linear, so these counts reproduce them exactly.

Runs kept: main block, no spillover (delta = 0), kappa = 10, beta1 - beta2 = 0.5
(beta2 = 0.5), vartheta = 0.05, every flood scenario and damage share.

Run:    python experiments/cascades_rnet/reduce_fcl_cells.py --raw-dir <folder with runs_chunk*.csv, fcl_chunk*.csv>
Writes: data/cascades_rnet/fcl_cells.csv.gz   (run_id, x, y, n_fcl)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

GZIP = {"method": "gzip", "mtime": 0}   # fixed timestamp: identical content gives identical files
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "cascades_rnet", "fcl_cells.csv.gz")


def map_runs(raw_dir):
    """The runs drawn in the maps, with their post-warm-up window."""
    runs = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(raw_dir, "runs_chunk*.csv"))))
    keep = (runs.success & (runs.block == "main") & np.isclose(runs.delta, 0.0)
            & np.isclose(runs.kappa, 10.0) & np.isclose(runs.beta2, 0.5) & np.isclose(runs.vartheta, 0.05))
    return runs.loc[keep, ["run_id", "warmup", "tot_step"]].set_index("run_id")


def count_cells(raw_dir, runs):
    """Post-warm-up foreclosures per (run, cell), reading the chunks in pieces."""
    counts = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "fcl_chunk*.csv"))):
        for part in pd.read_csv(path, usecols=["run_id", "step", "x", "y"], chunksize=2_000_000):
            part = part[part.run_id.isin(runs.index)].join(runs, on="run_id")
            part = part[(part.step >= part.warmup) & (part.step < part.tot_step)]
            counts.append(part.groupby(["run_id", "x", "y"]).size())
    return pd.concat(counts).groupby(level=[0, 1, 2]).sum().rename("n_fcl").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    a = ap.parse_args()
    runs = map_runs(a.raw_dir)
    cells = count_cells(a.raw_dir, runs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cells.to_csv(OUT, index=False, compression=GZIP)
    print(f"{len(runs)} runs, {cells.n_fcl.sum():,} foreclosures in {len(cells):,} (run, cell) rows -> {OUT}")


if __name__ == "__main__":
    main()
