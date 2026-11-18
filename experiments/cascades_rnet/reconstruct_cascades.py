"""Reconstruct the foreclosure cascades of the delta > 0 runs as weighted trees.

Every foreclosure is a root (it would have happened without the foreclosure discount, at
wtp0) or a child (attributed to the discount). A child at step t was caused by foreclosures at
step t - 1: a foreclosure discounts collateral within kappa, and U averages collateral within
kappa, so the reach is 2 kappa. Each child gets one parent, drawn among the step t - 1
foreclosures within 2 kappa with weight equal to the number of grid cells within kappa of
both the child and the candidate (the neighbours through which the candidate's discount
enters the child's U).

The size of a tree counts the root and all its descendants. Roots are kept in the window
[warmup, tot_step - 1 - 12], so that trees are not cut by the end of the run.

Second grouping, without weights or random draws: connected components of the candidate
graph, in which every child is joined to all its candidates. Components are kept when their
earliest member lies in the same window.

Run:    python experiments/cascades_rnet/reconstruct_cascades.py --raw-dir <folder with runs_chunk*.csv, fcl_chunk*.csv>
Writes: <out-dir>/sizes_by_run.csv.gz  run_id, method (tree or comp), size, count
        <out-dir>/trees_by_run.csv.gz  per run: parameters, foreclosures, attributed
                                                foreclosures, number of trees and components,
                                                sums of size, size^2, size^3, mean and skewness
"""
import argparse
import glob
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

GZIP = {"method": "gzip", "mtime": 0}   # fixed timestamp: identical content gives identical files
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "output", "cascades_rnet", "reduced")   # never data/: see REPRODUCE.md
CELL = ["setting", "lambda", "kappa", "beta2", "vartheta", "delta"]
TRUNC = 12        # roots in the last 12 months are dropped
RNG_SEED = 1      # parent draws of run r use the seed RNG_SEED + r
_OVERLAP = {}


def chunk_order(path):
    """Sort key of the chunk files: numbered chunks (fcl_chunk0.csv ... fcl_chunk12.csv, from the full
    design) in numeric order, then any other name (fcl_chunk_check.csv, from run.jl --run-ids) by name."""
    m = re.search(r"(\d+)\.csv$", path)
    return (0, int(m.group(1)), "") if m else (1, 0, os.path.basename(path))


def overlap_table(k):
    """W[dy + 2k, dx + 2k] = number of cells within k of both (0, 0) and (dx, dy)."""
    if k in _OVERLAP:
        return _OVERLAP[k]
    r = np.arange(-k, k + 1)
    yy, xx = np.meshgrid(r, r, indexing="ij")
    disk = (xx ** 2 + yy ** 2) <= k * k
    cy, cx = yy[disk], xx[disk]
    W = np.zeros((4 * k + 1, 4 * k + 1))
    for dy in range(-2 * k, 2 * k + 1):
        for dx in range(-2 * k, 2 * k + 1):
            if dx * dx + dy * dy <= 4 * k * k:
                W[dy + 2 * k, dx + 2 * k] = np.sum((cx - dx) ** 2 + (cy - dy) ** 2 <= k * k)
    _OVERLAP[k] = W
    return W


def build_forest(step, x, y, attr, k, rng):
    """Parent index per foreclosure (-1 for roots and orphans), orphan count, and the component
    label of every foreclosure in the candidate graph; arrays sorted by step."""
    W = overlap_table(k)
    parent = np.full(len(step), -1)
    uf = np.arange(len(step))

    def find(u):
        while uf[u] != u:
            uf[u] = uf[uf[u]]
            u = uf[u]
        return u

    starts = np.searchsorted(step, step - 1, side="left")
    ends = np.searchsorted(step, step - 1, side="right")
    orphans = 0
    for i in np.where(attr == 1)[0]:
        cand = np.arange(starts[i], ends[i])
        if len(cand):
            dx, dy = x[cand] - x[i], y[cand] - y[i]
            ok = dx * dx + dy * dy <= 4 * k * k
            cand, dx, dy = cand[ok], dx[ok], dy[ok]
        if len(cand) == 0:
            orphans += 1
            continue
        for j in cand:
            uf[find(i)] = find(j)
        w = W[dy + 2 * k, dx + 2 * k]
        parent[i] = cand[0] if len(cand) == 1 else rng.choice(cand, p=w / w.sum())
    comp = np.array([find(u) for u in range(len(step))])
    return parent, orphans, comp


def tree_sizes(parent):
    """Size of the subtree under every node (children always come after their parent)."""
    size = np.ones(len(parent), int)
    for i in range(len(parent) - 1, -1, -1):
        if parent[i] >= 0:
            size[parent[i]] += size[i]
    return size


def skew(S):
    S = np.asarray(S, float)
    return stats.skew(S) if len(S) > 2 and S.var() > 0 else np.nan


def process_run(args):
    meta, f, seed = args
    f = f.sort_values("step", kind="stable")
    step, x, y, attr = (f[c].values for c in ("step", "x", "y", "attr"))
    rng = np.random.default_rng(seed)
    parent, orphans, comp = build_forest(step, x, y, attr, int(round(meta["kappa"])), rng)
    size = tree_sizes(parent)
    lo, hi = int(meta["warmup"]), int(meta["tot_step"]) - 1
    inwin = (step >= lo) & (step <= hi)
    roots = (attr == 0) & (step >= lo) & (step <= hi - TRUNC)
    S = size[roots].astype(float)
    # components, kept when the earliest member is in [lo, hi - TRUNC]
    cdf = pd.DataFrame({"comp": comp, "step": step, "root": (attr == 0)})
    cg = cdf.groupby("comp").agg(size=("step", "size"), start=("step", "min"), n_roots=("root", "sum"))
    cg = cg[(cg.start >= lo) & (cg.start <= hi - TRUNC)]
    Sc = cg["size"].values.astype(float)
    n_in, n_attr = int(inwin.sum()), int((inwin & (attr == 1)).sum())
    hist = [(int(meta["run_id"]), m, int(v), int(c))
            for m, arr in (("tree", S), ("comp", Sc)) for v, c in zip(*np.unique(arr, return_counts=True))]
    return {**meta, "_hist": hist, "orphans": orphans, "n_fcl": n_in, "n_attr": n_attr,
            "R_gross": n_attr / n_in if n_in else np.nan,
            "n_trees": len(S), "s1": S.sum(), "s2": (S ** 2).sum(), "s3": (S ** 3).sum(),
            "tree_mean": S.mean() if len(S) else np.nan, "tree_skew": skew(S),
            "n_comp": len(Sc), "c1": Sc.sum(), "c2": (Sc ** 2).sum(), "c3": (Sc ** 3).sum(),
            "comp_mean": Sc.mean() if len(Sc) else np.nan, "comp_skew": skew(Sc),
            "roots_per_comp": cg.n_roots.mean() if len(cg) else np.nan,
            "n_trees_flooded_root": int((f.flooded.values[roots] > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", default=OUT_DIR, help="default: output/cascades_rnet/reduced")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    runs = pd.concat(pd.read_csv(p, float_precision="round_trip") for p in sorted(glob.glob(os.path.join(a.raw_dir, "runs_chunk*.csv"))))
    runs = runs[runs.success]
    key = ["setting", "lambda", "kappa", "beta2", "vartheta", "seed"]
    r0 = runs[runs.delta == 0][key + ["rate_tot"]].rename(columns={"rate_tot": "rate0"})
    runs = runs.merge(r0, on=key)
    runs["M"] = runs.rate_tot / runs.rate0       # foreclosure-rate multiplier of the run
    meta = runs.set_index("run_id", drop=False)[["run_id"] + CELL + ["seed", "seed_index", "warmup", "tot_step", "M"]]

    rows = []
    with Pool(a.workers) as pool:
        for path in sorted(glob.glob(os.path.join(a.raw_dir, "fcl_chunk*.csv")),
                           key=chunk_order):
            fcl = pd.read_csv(path, float_precision="round_trip")
            jobs = [(meta.loc[rid].to_dict(), g, RNG_SEED + int(rid))
                    for rid, g in fcl.groupby("run_id") if rid in meta.index and meta.loc[rid, "delta"] > 0]
            rows += pool.map(process_run, jobs, chunksize=8)
            print(f"{os.path.basename(path)}: {len(jobs)} runs", flush=True)

    sizes = pd.DataFrame([h for r in rows for h in r.pop("_hist")], columns=["run_id", "method", "size", "count"])
    sizes.to_csv(os.path.join(a.out_dir, "sizes_by_run.csv.gz"), index=False, compression=GZIP)
    trees = pd.DataFrame(rows)
    trees.to_csv(os.path.join(a.out_dir, "trees_by_run.csv.gz"), index=False, compression=GZIP)
    print(f"orphan children (no step t-1 foreclosure within 2 kappa): {int(trees.orphans.sum())} "
          f"of {int(trees.n_attr.sum())} attributed")


if __name__ == "__main__":
    main()
