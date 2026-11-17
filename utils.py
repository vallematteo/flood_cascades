"""Code shared by the figure, table and number scripts: paths, data loaders, constants,
statistics and the plot style.

Every loader reads one file of data/ and says which in its docstring.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

# ─── Paths

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
PREPARATION = os.path.join(ROOT, "preparation")
FIGURES = os.path.join(ROOT, "output", "figures")
TABLES = os.path.join(ROOT, "output", "tables")

# ─── Model constants

# Calibrated parametrisation: eta, m, varepsilon of the calibration (the parameter
# combination inside both empirical bands); everything else at its baseline value.
CALIBRATED = {"eta": 0.0382979672588408, "m": 5.246293367259204, "varepsilon": 0.0024974777754507}
WARMUP, TOT_STEP = 100, 400
YEARS = (TOT_STEP - WARMUP) / 12          # 25 post-warm-up years

# GEV parameters of the flood scenarios S1..S4: (location mu, scale sigma, shape xi).
GEV_SCENARIOS = {1: (10.0, 27.0, 0.1), 2: (15.789, 28.579, 0.5),
                 3: (21.579, 35.421, 0.8), 4: (28.421, 45.421, 1.2)}

# ─── Plot style

FS = 25                                    # base font size of every figure
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


# ─── Statistics

def t_interval(x, level=0.95):
    """Mean and half-width of the Student-t interval of the mean of x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (x.mean() if n else np.nan), np.nan
    return x.mean(), stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)


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


def max_t_threshold(ref, k, n_perm=2000, seed=0):
    """95% quantile, over random subsets of k rows of ref, of the largest KS distance between
    subset and ref across the parameters (max-T permutation threshold)."""
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        sub = ref[rng.choice(len(ref), k, replace=False)]
        null[b] = max(stats.ks_2samp(sub[:, j], ref[:, j]).statistic for j in range(ref.shape[1]))
    return np.quantile(null, 0.95)


def abc_ks(pos, accepted):
    """Per parameter: KS distance between retained and all combinations, the max-T threshold,
    whether the parameter is constrained (KS above the threshold), and retained quantiles."""
    p95 = max_t_threshold(pos.to_numpy(), len(accepted))
    rows = []
    for p in ABC_PARAMS:
        a, r = pos.loc[accepted, p], pos[p]
        q05, q50, q95 = a.quantile([0.05, 0.5, 0.95])
        rows.append({"param": p, "n_accepted": len(accepted), "ks": stats.ks_2samp(a, r).statistic,
                     "ks_p95": p95, "q05": q05, "q50": q50, "q95": q95})
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
    """Statistic (R_net or M_net) and R_gross per flooded scenario and delta, with the S0 run of the
    same seed and the contrast (lambda averaged within seed, then mean and t interval over seeds).
    Returns the flooded summary, the S0 summary and the per-seed table."""
    multiplier = stat == "M_net"
    d = ps[(ps.kappa == kappa) & (ps.beta2 == beta2)].assign(R_net=lambda x: x[stat])
    s0 = d[d.setting == 0][["delta", "seed", "R_net", "R_gross"]].rename(
        columns={"R_net": "R_net_S0", "R_gross": "R_gross_S0"})
    x = d[d.setting > 0].merge(s0, on=["delta", "seed"])
    x["dR_net"], x["dR_gross"] = contrast(x.R_net, x.R_net_S0, multiplier), x.R_gross - x.R_gross_S0
    cols = ["R_net", "R_gross", "R_net_S0", "R_gross_S0", "dR_net", "dR_gross"]
    w = x.groupby(["setting", "delta", "seed"])[cols].mean().reset_index()
    flooded = summarise(w, ["setting", "delta"], cols)
    if multiplier:
        flooded = contrast_to_percent(flooded, "dR_net")
    s0r = summarise(w[w.setting == 1].rename(columns={"R_net_S0": "Rn", "R_gross_S0": "Rg"}), ["delta"], ["Rn", "Rg"])
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
    s0 = s0r.rename(columns={"Rn": "R_net", "Rn_lo": "R_net_lo", "Rn_hi": "R_net_hi",
                             "Rg": "R_gross", "Rg_lo": "R_gross_lo", "Rg_hi": "R_gross_hi"})
    s0 = s0[s0.delta > 0].assign(x=lambda d: d.delta.map(pos))
    fl = flooded[flooded.delta > 0].assign(x=lambda d: d.delta.map(pos))
    return deltas, fl, s0


def delta_axis(ax, deltas):
    ax.set_xticks(range(len(deltas)), [f"{d:g}" for d in deltas])
    ax.set_xlim(-0.6, len(deltas) - 0.4)
