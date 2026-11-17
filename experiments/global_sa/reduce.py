"""Reduce the Parquet output of the sensitivity-analysis (or calibration) runs.

Only foreclosures and triggers from step 100 on (after the warm-up) enter.

Per run (runs_set<s>.csv.gz), with the run's sampled parameters:
    n_trig_fin, n_trig_strat     default triggers by channel (distress events)
    n_fcl_fin, n_fcl_strat, n_fcl_tot   foreclosures by channel
    mean_active, n_steps         mean number of active mortgages over the post-warm-up steps
    n_years                      complete post-warm-up years
    n_hhyr_fin, _strat, _any     households with a trigger in a year, summed over the years
    rate_tot, rate_fin, rate_strat     annual foreclosures per 100 active mortgages:
                                       100 * n_fcl / mean_active / (n_steps / 12)
    arrears_fin, _strat, _any    annual share of households in distress (%):
                                 100 * n_hhyr / (mean_active * n_years)

Per parameter vector, seeds pooled (vectors_set<s>.csv; sensitivity analysis only):
    n_financial, n_strategic, n_total   foreclosures per run, mean over the seeds
    perc_fin                     share of financial foreclosures, mean over the seeds of the
                                 per-run share (0 for a run without foreclosures)
    mean_months                  distress months per household that ever entered distress,
                                 over all households of all seeds

Run:    uv run python experiments/global_sa/reduce.py                     (sensitivity analysis)
        uv run python experiments/global_sa/reduce.py --run-list output/abc_calibration/run_list.csv \\
            --raw-dir output/abc_calibration/raw --out-dir data/abc_calibration --runs-only
Writes: <out-dir>/runs_set<s>.csv.gz and <out-dir>/vectors_set<s>.csv for every scenario s
"""
import argparse
import multiprocessing
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GZIP = {"method": "gzip", "mtime": 0}   # fixed timestamp: identical content gives identical files
WARMUP = 100
TOT_STEP = 400
ID_COLUMNS = ["run_id", "saltelli_sample_id", "sample_id", "setting", "seed_index", "seed"]


def read_run(raw_dir, run_id):
    """Post-warm-up triggers, foreclosures and active-mortgage series of one run."""
    sfx = f"_run{run_id:07d}"
    tr = pd.read_parquet(os.path.join(raw_dir, f"distress_data{sfx}.parquet"), columns=["step", "id", "default_reason"])
    fc = pd.read_parquet(os.path.join(raw_dir, f"default_data{sfx}.parquet"), columns=["step", "id", "default_reason"])
    md = pd.read_parquet(os.path.join(raw_dir, f"model_data{sfx}.parquet"), columns=["step", "tot_num_agents_active"])
    return tr[tr.step >= WARMUP], fc[fc.step >= WARMUP], md[md.step >= WARMUP]


def run_rates(tr, fc, md):
    n_years = (TOT_STEP - WARMUP) // 12
    yr = tr.assign(y=(tr.step - WARMUP) // 12)
    yr = yr[yr.y < n_years]                 # an incomplete last year is dropped
    fin = yr.default_reason == 1
    distinct = lambda d: len(d[["y", "id"]].drop_duplicates())
    n_steps = len(md)
    mean_active = float(md.tot_num_agents_active.mean()) if n_steps else np.nan
    rec = {
        "n_trig_fin": int((tr.default_reason == 1).sum()),
        "n_trig_strat": int((tr.default_reason != 1).sum()),
        "n_fcl_fin": int((fc.default_reason == 1).sum()),
        "n_fcl_strat": int((fc.default_reason != 1).sum()),
        "mean_active": mean_active, "n_steps": n_steps, "n_years": n_years,
        "n_hhyr_fin": distinct(yr[fin]), "n_hhyr_strat": distinct(yr[~fin]), "n_hhyr_any": distinct(yr),
    }
    rec["n_fcl_tot"] = rec["n_fcl_fin"] + rec["n_fcl_strat"]
    ok = n_steps > 0 and mean_active > 0
    for ch in ("tot", "fin", "strat"):
        rec[f"rate_{ch}"] = 100 * rec[f"n_fcl_{ch}"] / mean_active / (n_steps / 12) if ok else np.nan
    for ch in ("fin", "strat", "any"):
        rec[f"arrears_{ch}"] = 100 * rec[f"n_hhyr_{ch}"] / (mean_active * n_years) if ok and n_years else np.nan
    return rec


def reduce_vector(args):
    """All runs (seeds) of one parameter vector in one scenario."""
    runs, raw_dir = args
    run_recs, n_fin, n_strat, n_tot, shares = [], [], [], [], []
    months = []                              # distress months of each (seed, household)
    for run_id in runs.run_id:
        tr, fc, md = read_run(raw_dir, int(run_id))
        run_recs.append({"run_id": int(run_id), **run_rates(tr, fc, md)})
        r = fc.default_reason
        n_fin.append(int((r == 1).sum()))
        n_strat.append(int((r == -1).sum()))
        n_tot.append(len(r))
        shares.append(n_fin[-1] / n_tot[-1] if n_tot[-1] else 0.0)
        months.extend(tr.groupby("id").size().tolist())
    vector = {"n_strategic": float(np.mean(n_strat)), "n_financial": float(np.mean(n_fin)),
              "n_total": float(np.mean(n_tot)), "perc_fin": float(np.mean(shares)),
              "mean_months": float(np.mean(months)) if months else np.nan}
    return vector, run_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-list", default=os.path.join(ROOT, "output", "global_sa", "run_list.csv"))
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "output", "global_sa", "raw"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "global_sa"))
    ap.add_argument("--runs-only", action="store_true", help="per-run table only (calibration)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_NTASKS", os.cpu_count())))
    a = ap.parse_args()

    run_list = pd.read_csv(a.run_list, float_precision="round_trip")
    vector_id = "saltelli_sample_id" if "saltelli_sample_id" in run_list else "sample_id"
    # sampled parameters: those that vary across the run list (lambda and delta are fixed in the calibration)
    sampled = [c for c in run_list.columns if c not in ID_COLUMNS and run_list[c].nunique() > 1]
    # only the runs whose output is present (all runs, or the runs of a check)
    done = run_list.run_id.map(lambda r: os.path.exists(os.path.join(a.raw_dir, f"default_data_run{r:07d}.parquet")))
    run_list = run_list[done]
    os.makedirs(a.out_dir, exist_ok=True)

    for setting, sub in run_list.groupby("setting"):
        groups = [g for _, g in sub.groupby(vector_id)]
        with multiprocessing.Pool(a.workers) as pool:
            results = pool.map(reduce_vector, [(g, a.raw_dir) for g in groups])

        # per run, with its sampled parameters
        rates = pd.DataFrame([rec for _, recs in results for rec in recs])
        runs = sub[["run_id", "setting", "seed"] + sampled + [vector_id]].merge(rates, on="run_id")
        runs.sort_values("run_id").to_csv(os.path.join(a.out_dir, f"runs_set{setting}.csv.gz"),
                                          index=False, compression=GZIP)
        if not a.runs_only:
            vectors = pd.DataFrame([{vector_id: int(g[vector_id].iloc[0]), "setting": setting, **v}
                                    for g, (v, _) in zip(groups, results)])
            vectors.to_csv(os.path.join(a.out_dir, f"vectors_set{setting}.csv"), index=False)
        print(f"scenario {setting}: {len(groups)} vectors, {len(runs)} runs -> {a.out_dir}")


if __name__ == "__main__":
    main()
