"""Attribution-free clusters of foreclosures, the check on the cascade reconstruction.

The weighted trees of reconstruct_cascades.py rest on the within-month counterfactual wtp0
(root or child). This grouping drops it:
  * every foreclosure at step t is linked to every foreclosure at step t - 1 within 2 kappa
    (the reach of the spillover), whether or not it is attributed to the spillover;
  * clusters are the connected components of this graph, kept when their earliest member lies
    in [warmup, tot_step - 1 - 12] (the window of the trees);
  * the same grouping runs on the delta > 0 runs and on the delta = 0 runs. At delta = 0
    nothing causes anything, so its clusters measure clustering by coincidence.

Runs: kappa = 10, beta2 = 0.5 (beta1 - beta2 = 0.5), vartheta = 0.05, every scenario,
lambda and delta.

Run:    python experiments/cascades_rnet/cluster_attribution_free.py --raw-dir <folder with runs_chunk*.csv, fcl_chunk*.csv>
Writes: data/cascades_rnet/af_sizes_by_run.csv  run_id, size, count, method (= af)
        data/cascades_rnet/af_runs.csv          per run: foreclosures, clusters, largest cluster
"""
import argparse
import glob
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "cascades_rnet")
KAPPA, BETA2, VARTHETA = 10.0, 0.5, 0.05
TRUNC = 12


def clusters(step, x, y, k, lo, hi):
    """Component sizes of the graph linking every foreclosure at t to every one at t - 1 within
    2k; components kept when their earliest member is in [lo, hi]."""
    n = len(step)
    uf = np.arange(n)

    def find(u):
        while uf[u] != u:
            uf[u] = uf[uf[u]]
            u = uf[u]
        return u

    order = np.argsort(step, kind="stable")
    step, x, y = step[order], x[order], y[order]
    steps = np.unique(step)
    idx = {s: np.where(step == s)[0] for s in steps}
    for s in steps:
        if s - 1 not in idx:
            continue
        a, b = idx[s], idx[s - 1]
        tree = cKDTree(np.c_[x[b], y[b]])
        for i, nb in zip(a, tree.query_ball_point(np.c_[x[a], y[a]], r=2 * k)):
            for j in nb:
                ri, rj = find(i), find(b[j])
                if ri != rj:
                    uf[ri] = rj
    root = np.array([find(u) for u in range(n)])
    d = pd.DataFrame({"c": root, "step": step})
    g = d.groupby("c").agg(size=("step", "size"), start=("step", "min"))
    return g[(g.start >= lo) & (g.start <= hi)]["size"].values


def process_run(args):
    rid, k, lo, hi, f = args
    sz = clusters(f.step.values, f.x.values.astype(float), f.y.values.astype(float), k, lo, hi)
    v, c = np.unique(sz, return_counts=True)
    hist = [(rid, int(a), int(b)) for a, b in zip(v, c)]
    summary = dict(run_id=rid, n_fcl=int(((f.step >= lo) & (f.step <= hi)).sum()), n_clusters=len(sz),
                   largest=int(sz.max()) if len(sz) else 0)
    return hist, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    runs = pd.concat(pd.read_csv(p, usecols=["run_id", "success", "setting", "lambda", "kappa", "beta2", "vartheta",
                                              "delta", "seed", "warmup", "tot_step"])
                     for p in sorted(glob.glob(os.path.join(a.raw_dir, "runs_chunk*.csv"))))
    runs = runs[runs.success & np.isclose(runs.kappa, KAPPA) & np.isclose(runs.beta2, BETA2)
                & np.isclose(runs.vartheta, VARTHETA)].set_index("run_id", drop=False)

    hists, summaries = [], []
    with Pool(a.workers) as pool:
        for path in sorted(glob.glob(os.path.join(a.raw_dir, "fcl_chunk*.csv")),
                           key=lambda p: int(re.findall(r"(\d+)\.csv$", p)[0])):
            f = pd.read_csv(path, usecols=["run_id", "step", "x", "y"])
            f = f[f.run_id.isin(runs.index)]
            jobs = [(int(rid), int(round(runs.loc[rid, "kappa"])), int(runs.loc[rid, "warmup"]),
                     int(runs.loc[rid, "tot_step"]) - 1 - TRUNC, g) for rid, g in f.groupby("run_id")]
            for h, s in pool.map(process_run, jobs, chunksize=4):
                hists += h
                summaries.append(s)
            print(f"{os.path.basename(path)}: {len(jobs)} runs", flush=True)

    sizes = pd.DataFrame(hists, columns=["run_id", "size", "count"]).assign(method="af")
    sizes.to_csv(os.path.join(OUT, "af_sizes_by_run.csv"), index=False)
    per_run = pd.DataFrame(summaries).merge(
        runs[["run_id", "setting", "lambda", "delta", "seed"]].reset_index(drop=True), on="run_id")
    per_run.to_csv(os.path.join(OUT, "af_runs.csv"), index=False)
    print(f"{len(per_run)} runs -> af_sizes_by_run.csv, af_runs.csv in {OUT}")


if __name__ == "__main__":
    main()
