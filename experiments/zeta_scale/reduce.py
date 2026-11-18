"""Reduce the Parquet output of the zeta_scale runs to one row per run.

Only steps from 100 on (after the warm-up) enter, as in experiments/global_sa/reduce.py, and the
foreclosure rate and the distress share have the same definitions:
    acceptance        share of applicants admitted: admitted households / applicants, where every
                      month has rho * N_nj = 140 applicants (the population stays far below the
                      cap of 6,000 households, which would reduce the number of applicants)
    mean_active       mean number of active mortgages
    n_fcl_fin, n_fcl_strat, n_fcl_tot   foreclosures by channel
    n_hhyr_fin        households with a financial trigger in a year, summed over the 25 years
    rate_tot          annual foreclosures per 100 active mortgages: 100 * n_fcl_tot / mean_active / 25
    arrears_fin       annual share of households in financial distress (%):
                      100 * n_hhyr_fin / (mean_active * 25)
    median_age_fin    median loan age (months) at financial foreclosure

Run:    python experiments/zeta_scale/reduce.py
Writes: <out-dir>/runs.csv.gz (default output/zeta_scale/reduced/)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "model"))
from parameters import model_parameters  # noqa: E402
GZIP = {"method": "gzip", "mtime": 0}   # fixed timestamp: identical content gives identical files
_P = model_parameters("fixed")                      # model/parameters.csv
WARMUP, TOT_STEP = int(_P["warmup"]), int(_P["tot_step"])
APPLICANTS = int(_P["rho"] * _P["N_nj"])   # applicants per month (140)
MAX_AGENTS = 6000         # population cap of the model


def reduce_run(raw_dir, run_id):
    sfx = f"_run{run_id:07d}"
    tr = pd.read_parquet(os.path.join(raw_dir, f"distress_data{sfx}.parquet"), columns=["step", "id", "default_reason"])
    fc = pd.read_parquet(os.path.join(raw_dir, f"default_data{sfx}.parquet"), columns=["step", "t", "default_reason"])
    md = pd.read_parquet(os.path.join(raw_dir, f"model_data{sfx}.parquet"),
                         columns=["step", "tot_num_agents_active", "num_new_joiners"])
    tr, fc, md = tr[tr.step >= WARMUP], fc[fc.step >= WARMUP], md[md.step >= WARMUP]
    if md.tot_num_agents_active.max() > MAX_AGENTS - APPLICANTS:
        raise ValueError(f"run {run_id}: the population approaches the cap, applicants are not constant")

    n_years = (TOT_STEP - WARMUP) // 12
    fin_tr = tr[tr.default_reason == 1].assign(y=lambda d: (d.step - WARMUP) // 12)
    n_hhyr_fin = len(fin_tr[fin_tr.y < n_years][["y", "id"]].drop_duplicates())
    fin = fc.default_reason == 1
    mean_active = float(md.tot_num_agents_active.mean())
    n_fcl_tot = len(fc)
    return {
        "run_id": run_id,
        "acceptance": md.num_new_joiners.sum() / (APPLICANTS * len(md)),
        "mean_active": mean_active,
        "n_fcl_fin": int(fin.sum()), "n_fcl_strat": int((~fin).sum()), "n_fcl_tot": n_fcl_tot,
        "n_hhyr_fin": n_hhyr_fin,
        "rate_tot": 100 * n_fcl_tot / mean_active / (len(md) / 12),
        "arrears_fin": 100 * n_hhyr_fin / (mean_active * n_years),
        "median_age_fin": float(fc.t[fin].median()) if fin.any() else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-list", default=os.path.join(ROOT, "output", "zeta_scale", "run_list.csv"))
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "output", "zeta_scale", "raw"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output", "zeta_scale", "reduced"),
                    help="default: output/zeta_scale/reduced (never data/: see REPRODUCE.md)")
    a = ap.parse_args()

    run_list = pd.read_csv(a.run_list, float_precision="round_trip")
    reduced = pd.DataFrame([reduce_run(a.raw_dir, int(r)) for r in run_list.run_id])
    runs = run_list[["run_id", "zeta_scale", "seed_index", "seed"]].merge(reduced, on="run_id")
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "runs.csv.gz")
    runs.to_csv(out, index=False, compression=GZIP)
    print(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
