"""Code shared by the figure, table and number scripts: paths, data loaders, constants,
statistics and the plot style.

Every loader reads one file of data/ and says which in its docstring.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
