"""Code shared by the figure, table and number scripts: paths, data loaders, constants,
statistics and the plot style.

Every loader reads one file of data/ and says which in its docstring.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy import stats  # noqa: E402

# ─── Paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, os.path.join(ROOT, "model"))
from parameters import model_parameters  # noqa: E402
DATA = os.path.join(ROOT, "data")
PREPARATION = os.path.join(ROOT, "preparation")
FIGURES = os.path.join(ROOT, "output", "figures")
TABLES = os.path.join(ROOT, "output", "tables")

# ─── Model constants

# From model/parameters.csv. Calibrated parametrisation: eta, m, varepsilon of the calibration
# (the parameter combination inside both empirical bands); everything else at its baseline value.
FIXED, BASELINE, CALIBRATED = model_parameters("fixed"), model_parameters("baseline"), model_parameters("calibrated")
WARMUP, TOT_STEP = int(FIXED["warmup"]), int(FIXED["tot_step"])
YEARS = (TOT_STEP - WARMUP) / 12          # 25 post-warm-up years

# GEV parameters of the flood scenarios S1..S4: (location mu, scale sigma, shape xi).
GEV_SCENARIOS = {1: (10.0, 27.0, 0.1), 2: (15.789, 28.579, 0.5),
                 3: (21.579, 35.421, 0.8), 4: (28.421, 45.421, 1.2)}

# ─── Plot style

FS = 25                                    # base font size of every figure
# setting = flood scenario: 0 to 4 for S0 (no floods) to S4, as in the data files
SETTING_COLORS = {0: "#666666", 1: "#9ECAE1", 2: "#6BAED6", 3: "#3182BD", 4: "#08519C"}
CHANNEL_COLORS = {"fin": "#2166AC", "strat": "#B2182B"}


def scenario_label(s):
    return rf"$\mathcal{{S}}_{s}$"


def save_figure(fig, name, dpi=200):
    """Write output/figures/<name>.png and close the figure."""
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return path


# ─── Data loaders

def read(path):
    """CSV from data/, floats read exactly."""
    return pd.read_csv(os.path.join(DATA, path), float_precision="round_trip")


def load_cascade_runs():
    """data/cascades_rnet/runs_all.csv.gz: one row per run of the cascade experiment (65,520)."""
    return read("cascades_rnet/runs_all.csv.gz")


def load_gsa_vectors():
    """data/global_sa/vectors_set{0-4}.csv: per parameter vector and scenario, foreclosures per run
    by channel (n_financial, n_strategic, n_total), perc_fin and mean_months."""
    return pd.concat([read(f"global_sa/vectors_set{s}.csv") for s in range(5)], ignore_index=True)


def load_gsa_runs():
    """data/global_sa/runs_set{0-4}.csv.gz: one row per run of the sensitivity analysis (166,400)."""
    return pd.concat([read(f"global_sa/runs_set{s}.csv.gz") for s in range(5)], ignore_index=True)


def load_saltelli_matrix():
    """data/global_sa/design/saltelli_matrix.csv: the 3,328 parameter vectors."""
    return read("global_sa/design/saltelli_matrix.csv")


# ─── Calibration (rejection ABC at S0)

# Empirical bands (preparation/calibration_targets/): NHG foreclosure rate 2013-2016 and
# BKR share of households in financial distress 2014-2025, both in %.
EMPIRICAL_BANDS = {"rate_tot": (0.27, 0.42), "arrears_fin": (0.87, 3.54)}
ABC_PARAMS = ["kappa", "beta1", "beta2", "beta3", "eta", "vartheta", "m", "varepsilon", "exp_lb"]
ABC_TOLERANCES = [0.05, 0.01]      # share of combinations retained; 0.05 is the main tolerance
GREEK = {"kappa": r"\kappa", "beta1": r"\beta_1", "beta2": r"\beta_2", "beta3": r"\beta_3",
         "lambda": r"\lambda", "delta": r"\delta", "eta": r"\eta", "vartheta": r"\vartheta",
         "m": "m", "varepsilon": r"\varepsilon", "exp_lb": r"\mathrm{exp}_{lb}"}


def load_problem():
    """data/global_sa/design/problem_definition.csv as {parameter: (lower, upper)}."""
    p = read("global_sa/design/problem_definition.csv").set_index("parameter")
    return {k: (p.loc[k, "lower_bound"], p.loc[k, "upper_bound"]) for k in p.index}


def load_abc():
    """The calibration combinations (data/abc_calibration/design.csv) with the mean over their
    10 seeds of the foreclosure rate and the distress share (data/abc_calibration/runs_set0.csv.gz)."""
    design = read("abc_calibration/design.csv").set_index("sample_id")
    g = read("abc_calibration/runs_set0.csv.gz").groupby("sample_id")
    return design.join(g[list(EMPIRICAL_BANDS)].mean().join(g.size().rename("n_seeds")), how="inner")


def band_gap(x, lo, hi):
    """Distance of x to the interval [lo, hi] (0 inside)."""
    return np.where(x < lo, lo - x, np.where(x > hi, x - hi, 0.0))


def abc_distance(s):
    """Distance of every combination to the two bands, in log metrics scaled by their standard
    deviation over the combinations, and the tolerance d_tol of each retained share."""
    scale = {c: np.log(s[c]).std() for c in EMPIRICAL_BANDS}
    d = np.sqrt(sum((band_gap(np.log(s[c]), np.log(lo), np.log(hi)) / scale[c]) ** 2
                    for c, (lo, hi) in EMPIRICAL_BANDS.items()))
    return d, {t: np.quantile(d, t) for t in ABC_TOLERANCES}


def abc_position(s):
    """Position of every parameter in its sensitivity-analysis range (0 lower, 1 upper bound)."""
    b = load_problem()
    return pd.DataFrame({p: (s[p] - b[p][0]) / (b[p][1] - b[p][0]) for p in ABC_PARAMS}, index=s.index)


def max_t_null(ref, k, n_perm=2000, seed=0):
    """Null distribution of the max-T statistic: over random subsets of k rows of ref, the largest
    KS distance between subset and ref across the parameters."""
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        sub = ref[rng.choice(len(ref), k, replace=False)]
        null[b] = max(stats.ks_2samp(sub[:, j], ref[:, j]).statistic for j in range(ref.shape[1]))
    return null


def max_t_threshold(ref, k, n_perm=2000, seed=0):
    """95% quantile of the max-T null (max-T permutation threshold)."""
    return np.quantile(max_t_null(ref, k, n_perm, seed), 0.95)


def max_t_pvalue(ks, null):
    """Family-wise adjusted p-value of a KS statistic against the max-T null, with the
    finite-permutation correction of Phipson and Smyth (2010): (1 + #{null >= ks}) / (B + 1)."""
    return (1 + int(np.sum(null >= ks))) / (len(null) + 1)


def abc_ks(pos, accepted):
    """Per parameter: KS distance between retained and all combinations, the max-T threshold,
    the max-T adjusted p-value, whether the parameter is constrained (KS above the threshold),
    and retained quantiles."""
    null = max_t_null(pos.to_numpy(), len(accepted))
    p95 = np.quantile(null, 0.95)
    rows = []
    for p in ABC_PARAMS:
        a, r = pos.loc[accepted, p], pos[p]
        q05, q50, q95 = a.quantile([0.05, 0.5, 0.95])
        ks = stats.ks_2samp(a, r).statistic
        rows.append({"param": p, "n_accepted": len(accepted), "ks": ks, "ks_p95": p95,
                     "p_maxt": max_t_pvalue(ks, null), "q05": q05, "q50": q50, "q95": q95})
    out = pd.DataFrame(rows)
    out["constrained"] = out.ks > out.ks_p95
    return out


def abc_bootstrap(pos, accepted, ks_p95, n_boot=1000, seed=0):
    """Resample the combinations with replacement, d_tol fixed: 90% interval of the KS distance
    and of the retained 5% and 95% quantiles, and the share of resamples with KS above ks_p95."""
    rng = np.random.default_rng(seed)
    P = pos.to_numpy()
    acc = pos.index.isin(accepted)
    ks = np.empty((n_boot, P.shape[1]))
    q = np.empty((n_boot, P.shape[1], 2))
    for b in range(n_boot):
        i = rng.integers(0, len(P), len(P))
        Pb, ab = P[i], acc[i]
        for j in range(P.shape[1]):
            ks[b, j] = stats.ks_2samp(Pb[ab, j], Pb[:, j]).statistic
            q[b, j] = np.quantile(Pb[ab, j], [0.05, 0.95])
    return pd.DataFrame([{"param": p, "ks_lo": np.quantile(ks[:, j], 0.05), "ks_hi": np.quantile(ks[:, j], 0.95),
                          "p_above": (ks[:, j] > ks_p95).mean(),
                          "q05_lo": np.quantile(q[:, j, 0], 0.05), "q05_hi": np.quantile(q[:, j, 0], 0.95),
                          "q95_lo": np.quantile(q[:, j, 1], 0.05), "q95_hi": np.quantile(q[:, j, 1], 0.95)}
                         for j, p in enumerate(pos.columns)])


def abc_results():
    """Everything the calibration figure, table and numbers use: combinations s (with the
    distance), positions, retained combinations per tolerance, the combinations inside both
    bands, and the KS statistics per tolerance."""
    s = load_abc()
    s["abc_distance"], d_tol = abc_distance(s)
    pos = abc_position(s)
    accepted = {t: s.index[s.abc_distance <= d_tol[t]] for t in ABC_TOLERANCES}
    band = s.index[s.abc_distance == 0]
    ks = pd.concat([abc_ks(pos, accepted[t]).assign(tol=t) for t in ABC_TOLERANCES], ignore_index=True)
    return s, pos, accepted, band, ks


# ─── Foreclosure cascades

def load_paired_seed(vartheta=None):
    """data/cascades_rnet/paired_seed.csv.gz: per (cell, seed), the rates, R_net and the contrasts
    with the delta = 0 run and the S0 run of the same seed; adds the multiplier M_net = 1/(1 - R_net)."""
    ps = read("cascades_rnet/paired_seed.csv.gz")
    assert (ps.R_net < 1).all(), "R_net = 1 somewhere: multiplier infinite"
    ps["M_net"] = 1 / (1 - ps.R_net)
    return ps if vartheta is None else ps[ps.vartheta == vartheta]


def summarise(df, keys, cols):
    """Per group of keys: mean of every column with its 95% Student-t interval (columns c, c_lo, c_hi)."""
    rows = []
    for k, g in df.groupby(keys):
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for c in cols:
            x = np.asarray(g[c], float)            # missing values propagate (nan mean)
            m = x.mean()
            h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            rec[c], rec[c + "_lo"], rec[c + "_hi"] = m, m - h, m + h
        rows.append(rec)
    return pd.DataFrame(rows)


def contrast(a, b, multiplier):
    """Flooded against S0 of the same seed: difference for R_net, log ratio for the multiplier."""
    return np.log(a / b) if multiplier else a - b


def contrast_to_percent(df, col):
    """Mean log ratio and its interval as a percentage change, 100 (exp(.) - 1)."""
    for c in (col, col + "_lo", col + "_hi"):
        df[c] = 100 * np.expm1(df[c])
    return df


def r_by_scenario(ps, stat="M_net", kappa=10, beta2=0.5):
    """Statistic stat (the net reproduction number R_net or the multiplier M_net) and R_gross per
    flooded scenario and delta, with the S0 run of the same seed and the contrast (lambda averaged
    within seed, then mean and t interval over seeds). Columns are named after the statistic:
    stat, stat_S0 and d<stat> (difference for R_net, % change for M_net), and the same for R_gross.
    Returns the flooded summary, the S0 summary (columns stat and R_gross) and the per-seed table."""
    multiplier = stat == "M_net"
    d = ps[(ps.kappa == kappa) & (ps.beta2 == beta2)]
    s0 = d[d.setting == 0][["delta", "seed", stat, "R_gross"]].rename(
        columns={stat: f"{stat}_S0", "R_gross": "R_gross_S0"})
    x = d[d.setting > 0].merge(s0, on=["delta", "seed"])
    x[f"d{stat}"], x["dR_gross"] = contrast(x[stat], x[f"{stat}_S0"], multiplier), x.R_gross - x.R_gross_S0
    cols = [stat, "R_gross", f"{stat}_S0", "R_gross_S0", f"d{stat}", "dR_gross"]
    w = x.groupby(["setting", "delta", "seed"])[cols].mean().reset_index()
    flooded = summarise(w, ["setting", "delta"], cols)
    if multiplier:
        flooded = contrast_to_percent(flooded, f"d{stat}")
    # every flooded scenario carries the same S0 runs; take them once, from S1
    s0r = summarise(w.loc[w.setting == 1, ["delta", "seed", f"{stat}_S0", "R_gross_S0"]]
                    .rename(columns={f"{stat}_S0": stat, "R_gross_S0": "R_gross"}), ["delta"], [stat, "R_gross"])
    return flooded, s0r, w


CASCADE_GROUPS = {"S0": [0], "S1": [1], "S4": [4]}


def load_cascade_sizes(kappa=10, beta2=0.5, vartheta=0.05):
    """Cascade sizes (data/cascades_rnet/sizes_by_run.csv.gz: run_id, method, size, count) of the
    runs of one slice, and their run parameters (trees_by_run.csv.gz)."""
    sizes = read("cascades_rnet/sizes_by_run.csv.gz")
    meta = read("cascades_rnet/trees_by_run.csv.gz")[
        ["run_id", "setting", "lambda", "kappa", "beta2", "vartheta", "delta", "seed"]]
    meta = meta[(meta.kappa == kappa) & (meta.beta2 == beta2) & (meta.vartheta == vartheta)]
    return sizes[sizes.run_id.isin(meta.run_id)], meta


def size_density(sizes_all, meta, bw=0.3, smax=1500, n_grid=600, groups=None):
    """Share of foreclosures by cascade size, per method (tree, comp), scenario group and delta,
    smoothed with a Gaussian kernel of bandwidth bw in log size (each cascade weighted by its
    size). Returns (grid, dens, peaks): dens[(method, group, delta)] is the density on the log-size
    grid; peaks has, per cell, the peak of the large-cascade mode (sweep_peak, sizes > 2), the
    valley between it and size 1, and the share of foreclosures in cascades of size >= the valley
    (sweep_share, counted on the raw sizes)."""
    grid = np.linspace(0, np.log(smax), n_grid)
    groups = groups or CASCADE_GROUPS
    s = sizes_all.merge(meta[["run_id", "setting", "delta"]], on="run_id")
    s = pd.concat([s[s.setting.isin(v)].assign(group=k) for k, v in groups.items()])
    dens, rows = {}, []
    for (m, fl, d), g in s.groupby(["method", "group", "delta"]):
        x, w = np.log(g["size"].values), (g["size"] * g["count"]).values.astype(float)
        k = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2) @ w
        k /= k.sum()
        dens[(m, fl, d)] = k
        loc = [j for j in range(1, len(k) - 1) if k[j] > k[j - 1] and k[j] >= k[j + 1] and np.exp(grid[j]) > 2]
        pk = val = np.nan
        share = 0.0
        if loc:
            j = max(loc, key=lambda j: k[j])
            v = int(np.argmin(k[:j + 1]))
            pk, val = np.exp(grid[j]), np.exp(grid[v])
            share = w[g["size"].values >= val].sum() / w.sum()
        rows.append(dict(method=m, group=fl, delta=d, sweep_peak=pk, valley=val, sweep_share=share,
                         n_runs=g.run_id.nunique(), n_cascades=int(g["count"].sum())))
    return grid, dens, pd.DataFrame(rows)


# ─── Plot helpers of the cascade figures

def style_axes(ax, xlabel, ylabel, zero=False, fs=FS):
    ax.set_xlabel(xlabel, fontsize=fs, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=fs, fontweight="bold")
    ax.tick_params(labelsize=fs - 5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if zero:
        ax.axhline(0, color="black", linewidth=1.5)


def points(ax, d, x, y, color, label, dx=0.0, marker="o", **kw):
    """Means as points with 95% interval error bars, shifted by dx along x."""
    d = d.sort_values(x)
    yerr = [d[y] - d[f"{y}_lo"], d[f"{y}_hi"] - d[y]] if f"{y}_lo" in d else None
    ax.errorbar(d[x] + dx, d[y], yerr=yerr, fmt=marker, markersize=11, color=color, label=label,
                elinewidth=2, capsize=4, linestyle="none", **kw)


def curve(ax, s, col, color, ls="-", marker="o"):
    """Mean line against delta with its 95% band."""
    s = s.sort_values("delta")
    ax.plot(s.delta, s[col], color=color, linestyle=ls, marker=marker, markersize=9, linewidth=3)
    ax.fill_between(s.delta, s[col + "_lo"], s[col + "_hi"], color=color, alpha=0.18, linewidth=0)


def level_axis(ax, multiplier):
    """Level panels: log axis for the multiplier, line at the no-cascade level (R_net 0, M 1)."""
    if multiplier:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.axhline(1.0 if multiplier else 0.0, color="black", linewidth=1.5)


def delta_positions(flooded, s0r):
    """Categorical x positions of the delta > 0 cells, for the flooded and the S0 summaries."""
    deltas = sorted(flooded.delta[flooded.delta > 0].unique())
    pos = {d: i for i, d in enumerate(deltas)}
    s0 = s0r[s0r.delta > 0].assign(x=lambda d: d.delta.map(pos))
    fl = flooded[flooded.delta > 0].assign(x=lambda d: d.delta.map(pos))
    return deltas, fl, s0


def delta_axis(ax, deltas):
    ax.set_xticks(range(len(deltas)), [f"{d:g}" for d in deltas])
    ax.set_xlim(-0.6, len(deltas) - 0.4)


# ─── Sensitivity analysis: Sobol' and PAWN indices

GSA_SOBOL_SEED = 1        # seed of SALib's bootstrap of the Sobol' confidence intervals
PAWN_S = 10               # conditioning intervals of PAWN
PAWN_PERMUTATIONS, PAWN_PERMUTATION_SEED = 10_000, 12345


def gsa_problem():
    """SALib problem of the sensitivity analysis (names and bounds of the 11 parameters)."""
    p = read("global_sa/design/problem_definition.csv")
    return {"num_vars": len(p), "names": p.parameter.tolist(),
            "bounds": [[r.lower_bound, r.upper_bound] for r in p.itertuples()]}


def gsa_outcome(setting, Y):
    """Outcome Y of every parameter vector in one scenario, in design order; missing values
    (vectors without foreclosures) are replaced by the mean."""
    y = read(f"global_sa/vectors_set{setting}.csv").sort_values("saltelli_sample_id")[Y].to_numpy(float)
    return np.where(np.isnan(y), np.nanmean(y), y) if np.isnan(y).any() else y


def sobol_indices(Y):
    """First- and total-order Sobol' indices of outcome Y in every scenario, with 95% bootstrap
    intervals: {setting: SALib result}."""
    from SALib.analyze import sobol
    problem = gsa_problem()
    D = problem["num_vars"]
    out = {}
    for s in range(5):
        y = gsa_outcome(s, Y)
        n = len(y) // (D + 2)
        # the design is stored in blocks [A; B; AB_0; ...; AB_{D-1}]; SALib expects each base
        # sample's rows together, [A_i, AB_0_i, ..., AB_{D-1}_i, B_i]
        y_interleaved = np.column_stack([y[0:n], *[y[(j + 2) * n:(j + 3) * n] for j in range(D)],
                                         y[n:2 * n]]).flatten()
        out[s] = sobol.analyze(problem, y_interleaved, calc_second_order=False, print_to_console=False,
                               seed=GSA_SOBOL_SEED)
    return out


def pawn_design():
    """Parameter values of the A and B blocks of the Saltelli design (2 x 256 independent draws),
    the sample PAWN uses."""
    problem = gsa_problem()
    n = len(load_saltelli_matrix()) // (problem["num_vars"] + 2)
    m = load_saltelli_matrix().sort_values("saltelli_sample_id")
    return m[m.saltelli_sample_id < 2 * n][problem["names"]].to_numpy(float)


def pawn_indices(Y, S=PAWN_S):
    """PAWN indices of outcome Y in every scenario (median, minimum and maximum over the S
    conditioning intervals): {setting: SALib result}."""
    from SALib.analyze import pawn
    problem, X = gsa_problem(), pawn_design()
    return {s: pawn.analyze(problem, X, gsa_outcome(s, Y)[:len(X)], S=S, seed=0) for s in range(5)}


def _pawn_medians(labels, n_values, full_cdf, masks):
    """Median over the conditioning intervals of the KS distance, per parameter (SALib's statistic)."""
    medians = np.empty(len(masks))
    for j, parameter_masks in enumerate(masks):
        distances = []
        for mask in parameter_masks:
            n_selected = int(mask.sum())
            if n_selected:
                cdf = np.cumsum(np.bincount(labels[mask], minlength=n_values)) / n_selected
                distances.append(np.max(np.abs(cdf - full_cdf)))
        medians[j] = np.median(distances)
    return medians


def pawn_threshold(Y, S=PAWN_S, n_permutations=PAWN_PERMUTATIONS, seed=PAWN_PERMUTATION_SEED):
    """95% max-T permutation threshold of the PAWN indices of outcome Y: the outcome is permuted
    across the design (the same permutation in every scenario), and the threshold is the 95%
    quantile of the largest index over the 11 parameters and 5 scenarios."""
    X = pawn_design()
    masks = []
    for j in range(X.shape[1]):
        q = np.nanquantile(X[:, j], np.arange(S + 1) / S)
        masks.append([(X[:, j] >= q[s]) & (X[:, j] < q[s + 1]) for s in range(S)])
    prepared = []
    for s in range(5):
        _, labels, counts = np.unique(gsa_outcome(s, Y)[:len(X)], return_inverse=True, return_counts=True)
        prepared.append((labels, len(counts), np.cumsum(counts) / len(X)))
    rng = np.random.default_rng(seed)
    largest = np.empty(n_permutations)
    for b in range(n_permutations):
        perm = rng.permutation(len(X))
        largest[b] = max(_pawn_medians(lab[perm], nv, cdf, masks).max() for lab, nv, cdf in prepared)
    return float(np.quantile(largest, 0.95))


def _index_axes(ax, ylabel, exp_lb=r"\mathrm{exp}_{lb}"):
    names = gsa_problem()["names"]
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([f"${exp_lb if p == 'exp_lb' else GREEK[p]}$" for p in names], fontsize=FS)
    ax.tick_params(axis="y", labelsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS, fontweight="bold")
    ax.set_xlabel("Parameters", fontsize=FS, fontweight="bold")
    ax.grid(axis="y", linestyle="-", alpha=0.2, color="gray", linewidth=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.legend(framealpha=0, fontsize=FS - 5, loc="upper right", edgecolor="black", fancybox=False)
    ax.set_ylim(-0.05, 1.05)


def sobol_figure(Y, order, name):
    """Sobol' indices of outcome Y (order 'S1' first-order or 'ST' total-order) per parameter and
    scenario, with 95% bootstrap intervals; written to output/figures/<name>.png."""
    indices = sobol_indices(Y)
    x, offsets = np.arange(len(gsa_problem()["names"])), np.linspace(-0.32, 0.32, 5)
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (s, si) in enumerate(indices.items()):
        ax.errorbar(x + offsets[i], si[order], yerr=si[f"{order}_conf"], fmt="o", color=SETTING_COLORS[s],
                    capsize=6, markersize=10, linewidth=2.5, label=f"$\\mathcal{{S}}_{{{s}}}$", zorder=3, elinewidth=2)
    _index_axes(ax, "First-Order Sobol Index ($S_1$)" if order == "S1" else "Total-Order Sobol Index ($S_T$)")
    fig.tight_layout()
    save_figure(fig, name, dpi=300)


def pawn_figure(Y, name):
    """Median PAWN index of outcome Y per parameter and scenario, bars from the minimum to the
    maximum over the conditioning intervals, dotted line at the 95% max-T permutation threshold;
    written to output/figures/<name>.png."""
    indices, threshold = pawn_indices(Y), pawn_threshold(Y)
    x, offsets = np.arange(len(gsa_problem()["names"])), np.linspace(-0.32, 0.32, 5)
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (s, si) in enumerate(indices.items()):
        med = np.asarray(si["median"], float)
        err = np.vstack([np.clip(med - np.asarray(si["minimum"], float), 0, None),
                         np.clip(np.asarray(si["maximum"], float) - med, 0, None)])
        ax.errorbar(x + offsets[i], med, yerr=err, fmt="o", color=SETTING_COLORS[s], capsize=6, markersize=10,
                    linewidth=2.5, label=f"$\\mathcal{{S}}_{{{s}}}$", zorder=3, elinewidth=2)
    ax.axhline(threshold, color="#b2182b", linewidth=2.0, linestyle=":", alpha=0.95, label="95% perm.", zorder=2)
    _index_axes(ax, "PAWN Index (median)", exp_lb=r"\mathit{exp}_{lb}")
    fig.tight_layout()
    save_figure(fig, name, dpi=300)


# ─── Foreclosure timing

TIMING_CHANNELS = {1: ("financial", "#0b7285"), -1: ("strategic", "#b2182b")}


def load_timing(design):
    """One row per post-warm-up foreclosure of the foreclosure-timing design 'gap' or 'beta3'
    (data/foreclosure_timing/<design>_foreclosures.csv.gz), with the parameters of its run
    (<design>_runs.csv) and gap = beta1 - beta2."""
    fcl = read(f"foreclosure_timing/{design}_foreclosures.csv.gz")
    runs = read(f"foreclosure_timing/{design}_runs.csv")
    for k, v in CALIBRATED.items():
        assert np.allclose(runs[k], v), k
    d = fcl.merge(runs[["run_id", "setting", "seed", "beta1", "beta2", "beta3", "lambda", "delta", "warmup"]],
                  on="run_id", how="left", validate="many_to_one")
    d["gap"] = (d.beta1 - d.beta2).round(2)
    return d


def style_density_axes(ax, xlabel, xmax):
    ax.set_xlabel(xlabel, fontsize=FS, fontweight="bold")
    ax.set_ylabel("Density", fontsize=FS, fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.tick_params(labelsize=FS - 5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def sparks_and_sweeps(sizes_all, meta, peaks, groups=None):
    """Per method, scenario group and delta: cascades per run, large cascades per run and per
    1,000 cascades. A large cascade is one of size >= the valley of its cell (peaks of size_density)."""
    groups = groups or CASCADE_GROUPS
    s = sizes_all.merge(meta[["run_id", "setting", "delta"]], on="run_id")
    s = pd.concat([s[s.setting.isin(v)].assign(group=k) for k, v in groups.items()])
    s = s.merge(peaks[["method", "group", "delta", "valley"]], on=["method", "group", "delta"], how="left")
    s["sweep"] = (s["count"] * ((s["size"] >= s.valley) & s.valley.notna())).astype(float)
    g = s.groupby(["method", "group", "delta"])
    out = pd.DataFrame({"runs": g.run_id.nunique(), "cascades": g["count"].sum(), "sweeps": g.sweep.sum()}).reset_index()
    out["sparks_per_run"] = out.cascades / out.runs
    out["sweeps_per_run"] = out.sweeps / out.runs
    out["sweeps_per_1000_sparks"] = 1000 * out.sweeps / out.cascades
    return out


def load_attribution_free():
    """Attribution-free clusters (data/cascades_rnet/af_sizes_by_run.csv: run_id, size, count) and
    their runs (af_runs.csv: kappa = 10, beta1 - beta2 = 0.5, vartheta = 0.05)."""
    return read("cascades_rnet/af_sizes_by_run.csv"), read("cascades_rnet/af_runs.csv")


def compare_methods(n_boot=2000, seed=1):
    """Share of all foreclosures in large cascades (size >= the valley of the cell, each method
    with its own valley), pooled over the runs of a cell, for weighted trees (tree), connected
    components (comp) and attribution-free clusters (af), with a 95% interval from resampling
    seeds; plus the peak of the large-cascade mode."""
    af_sizes, af_runs = load_attribution_free()
    meta = af_runs[["run_id", "setting", "lambda", "delta", "seed"]]
    _, _, af_peaks = size_density(af_sizes, meta)
    sizes, tree_meta = load_cascade_sizes()
    _, _, tree_peaks = size_density(sizes, tree_meta)
    modes = pd.concat([tree_peaks, af_peaks])
    att = read("cascades_rnet/sizes_by_run.csv.gz")
    att = att[att.run_id.isin(meta.run_id)]
    sizes_all = pd.concat([att[["run_id", "method", "size", "count"]], af_sizes[["run_id", "method", "size", "count"]]])

    rng = np.random.default_rng(seed)
    s = sizes_all.merge(meta[["run_id", "setting", "delta", "seed"]], on="run_id")
    s = pd.concat([s[s.setting.isin(v)].assign(group=k) for k, v in CASCADE_GROUPS.items()])
    s = s.merge(modes[["method", "group", "delta", "valley"]], on=["method", "group", "delta"], how="left")
    s["fc"] = s["size"] * s["count"]
    s["fc_sweep"] = s.fc * ((s["size"] >= s.valley) & s.valley.notna())
    sd = s.groupby(["method", "group", "delta", "seed"])[["fc", "fc_sweep"]].sum().reset_index()
    out = []
    for (m, g, d), x in sd.groupby(["method", "group", "delta"]):
        a, b = x.fc_sweep.values, x.fc.values
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        bs = a[idx].sum(1) / b[idx].sum(1)
        out.append(dict(method=m, group=g, delta=d, share=a.sum() / b.sum(), share_lo=np.quantile(bs, 0.025),
                        share_hi=np.quantile(bs, 0.975), n_seeds=len(a)))
    return pd.DataFrame(out).merge(modes[["method", "group", "delta", "sweep_peak"]],
                                   on=["method", "group", "delta"], how="left")



# ─── Maps of SI Fig. 2

MAP_BANDWIDTH = 10.0       # kernel standard deviation of the foreclosure maps, in grid cells
# flood maps from the smallest to the largest extent
TIERS = [("Minor", "grote.png"), ("Moderate", "middel.png"), ("Severe", "kleine.png"), ("Catastrophic", "extreme_kleine.png")]

def read_map(name):
    """Same transform as read_map in model/utils.jl: invert grey, scale to [0, 1]."""
    a = np.asarray(Image.open(os.path.join(PREPARATION, "flood_maps", name)).convert("L"), dtype=float) / 255
    a = 1.0 - a
    return a / a.max() if a.max() > 0 else a


def flood_bands():
    """Number of tiers flooding each cell (0..4), after checking the extents are nested."""
    ext = [read_map(f) > 0 for _, f in TIERS]
    for (n_small, _), (n_big, _), a, b in zip(TIERS, TIERS[1:], ext, ext[1:]):
        bad = int((a & ~b).sum())
        if bad:
            raise SystemExit(f"Extents not nested: {bad} {n_small} cells outside {n_big}")
    return sum(e.astype(int) for e in ext)


def kde_grid(rows, cols, shape, bw, weights=None):
    """Gaussian KDE on the grid: point histogram convolved with N(0, bw^2 I), reflecting
    boundary, so the estimate integrates to the number (or weight) of points."""
    h = np.zeros(shape)
    np.add.at(h, (rows, cols), 1.0 if weights is None else weights)
    return gaussian_filter(h, sigma=bw, mode="reflect", truncate=4.0)


def rate_map(runs, fcl, shape, pop_share, bw):
    """Local annual foreclosure rate (%), defined as rate_tot in the reduced data
    (experiments/global_sa/reduce.py): per run, 100 * foreclosures / mean_active / years,
    then the mean over seeds. Locally, foreclosures are the run's KDE and mean_active is
    the run's mean active population times the cell's share of the smoothed placement
    weight. The mean over runs is one KDE with each point weighted by
    100 / (n_runs * mean_active_run * years_run)."""
    w_run = 100 / (len(runs) * runs.mean_active * (runs.tot_step - runs.warmup) / 12)
    w = fcl.run_id.map(w_run.set_axis(runs.run_id)).to_numpy()
    num = kde_grid(fcl.y.values - 1, fcl.x.values - 1, shape, bw, weights=w)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = num / pop_share
    # Check: averaging the map over the population gives back the seed-mean rate_tot.
    print(f"  population-weighted mean of map {np.nansum(rate * pop_share):.3f} %/yr "
          f"vs seed-mean rate_tot {runs.rate_tot.mean():.3f} %/yr")
    return rate


def load_map_runs(flood_setting):
    """The runs drawn in the maps (S0 and flood_setting: cascade experiment, main block, delta = 0,
    kappa = 10, beta1 - beta2 = 0.5, vartheta = 0.05) and their foreclosures per cell."""
    r = load_cascade_runs()
    r = r[r.success & (r.block == "main") & np.isclose(r.delta, 0.0) & np.isclose(r.kappa, 10.0)
          & np.isclose(r.beta2, 0.5) & np.isclose(r.vartheta, 0.05)]
    cells = read("cascades_rnet/fcl_cells.csv.gz")
    return {s: (r[r.setting == s], cells[cells.run_id.isin(r.run_id[r.setting == s])]) for s in (0, flood_setting)}


def map_fields(flood_setting=4, min_pop=0.05):
    """Housing density, region, flood bands, local foreclosure rate at S0 and at flood_setting,
    and their log2 ratio, masked where the smoothed housing density is below min_pop of its maximum."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    housing = read_map("nvm_cleaned.png")
    region = binary_fill_holes(binary_closing(housing > 0, iterations=3))
    band = np.where(region, flood_bands(), 0)
    pop_share = kde_grid(*np.nonzero(housing >= 0), housing.shape, MAP_BANDWIDTH, weights=housing[housing >= 0])
    pop_share /= pop_share.sum()
    fields = {}
    for s, (runs, cells) in load_map_runs(flood_setting).items():
        fcl = cells.loc[cells.index.repeat(cells.n_fcl)]          # one row per foreclosure
        fields[s] = rate_map(runs, fcl, housing.shape, pop_share, MAP_BANDWIDTH)
    keep = region & (pop_share >= min_pop * pop_share.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(keep, np.log2(fields[flood_setting] / fields[0]), np.nan)
        rate0 = np.where(keep, fields[0], np.nan)
        rate_flood = np.where(keep, fields[flood_setting], np.nan)
    ratio[~np.isfinite(ratio)] = np.nan
    return housing, region, band, rate0, rate_flood, ratio


# ─── Admission of applicants (SI Table 5)

ADMISSION_DRAWS = 2_000_000
ADMISSION_SEED = 0
ACCEPT_SCORE = FIXED["acpt_score"]   # admission threshold A
LOAN_TERM = 120             # months, n_{i,0} of entrants


def rand_burr(rng, c, d, loc, scale, n):
    """Burr Type III draws by inverse transform, as rand_burr in model/model.jl."""
    u = rng.uniform(0.0, 1.0, n)
    return scale * (u ** (-1.0 / d) - 1.0) ** (-1.0 / c) + loc


def scorecard(zeta, income, sen, ltv, install):
    """Admission score, as scorecard in model/utils.jl (SI Table 4)."""
    cap = np.select([zeta <= 0.4, zeta <= 0.6, zeta <= 0.8], [20, 40, 60], 80)
    inc = np.select([income <= 1500, income <= 2000, income <= 2500, income <= 3000], [13, 26, 39, 52], 65)
    snr = np.select([sen <= 100, sen <= 200, sen <= 300, sen <= 400], [13, 26, 39, 52], 65)
    ltv_s = np.select([ltv <= 0.4, ltv <= 0.6, ltv <= 0.8], [50, 37.5, 25], 12.5)
    inst = np.select([install <= 800, install <= 1000, install <= 1200], [40, 30, 20], 10)
    return cap + inc + snr + ltv_s + inst


def draw_applicants(n=ADMISSION_DRAWS, seed=ADMISSION_SEED, zeta_scale=FIXED["zeta_scale"]):
    """Applicants at entry with the distributions of the model (add_agents! in model/model.jl,
    SI Table 3), their score and admission. Property value v in thousand euro. The parameters
    are those of the constants at the top of model/model.jl (INCOME_BURR, SENIORITY_BETA, ...):
    change both together."""
    rng = np.random.default_rng(seed)
    a = pd.DataFrame({
        "income": rand_burr(rng, 3.30, 0.45, -12.76, 3101.46, n),
        "sen": rng.beta(0.88, 2.79, n) * 106.34 + 1.0,
        "zeta": rand_burr(rng, 9.42, 0.14, -0.11, 73.40, n) * zeta_scale,
        "ltv": np.abs(rng.normal(0.57, 0.1, n)),
        "v": rand_burr(rng, 3.38, 0.92, -1.34, 195.25, n),
    })
    a["install"] = a.ltv * a.v * 1000 / LOAN_TERM
    a["install_share"] = a.install / a.income
    a["score"] = scorecard(a.zeta, a.income, a.sen, a.ltv, a.install)
    a["admitted"] = a.score >= ACCEPT_SCORE
    return a
