"""Every number quoted in the paper and the SI, recomputed from data/ and compared with the
value printed in the text.

Each number is a row: where it appears, what it is, the value computed here, the value
printed in the paper, and the number of decimals it is printed with. A row matches when the
computed value rounded to those decimals equals the printed value. Sections follow the paper:
initialization, calibration, Results (foreclosure rates, composition, cascades, timing),
sensitivity analysis, cascade method and foreclosure timing of the SI.

Run:    uv run python paper_numbers.py
Writes: output/numbers.csv; prints every row and a summary; exits with an error if any
        number does not match.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from utils import (ABC_PARAMS, CALIBRATED, EMPIRICAL_BANDS, PREPARATION, ROOT, YEARS, abc_bootstrap, abc_distance,
                   abc_ks, abc_position, abc_results, compare_methods, gsa_problem, load_abc, load_cascade_runs,
                   load_cascade_sizes, load_gsa_runs, load_gsa_vectors, load_paired_seed, load_problem,
                   load_saltelli_matrix, load_timing, pawn_indices, pawn_threshold, r_by_scenario, read,
                   size_density, sobol_indices, sparks_and_sweeps, summarise)

ROWS = []


def number(where, what, computed, printed, decimals=2):
    """Record one quoted number. printed=None records a value the text uses without printing it."""
    computed = float(computed)
    ok = None if printed is None else bool(np.isclose(round(computed, decimals), printed, rtol=0, atol=1e-9))
    ROWS.append(dict(where=where, what=what, computed=computed, printed=printed, decimals=decimals, match=ok))


def t_mean(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


# ─── SI, Baseline parametrisation (initialization) ───────────────────────────────────────

def initialization():
    b = pd.read_csv(os.path.join(PREPARATION, "initialization", "baseline_parameters.csv")).set_index("quantity").value
    w = "SI Baseline parametrisation"
    number(w, "decile-years in the fit (n)", b.n_decile_years, 30, 0)
    number(w, "lowest monthly income (EUR)", b.income_min_eur_month, 550, 0)
    number(w, "m (NLS)", b.m, 6.22)
    number(w, "s.e. of m", b.se_m, 0.79)
    number(w, "varepsilon per EUR (x1e3)", 1e3 * b.varepsilon, 1.98)
    number(w, "s.e. of varepsilon per EUR (x1e4)", 1e4 * b.se_varepsilon, 3.61)
    number(w, "R^2 of the NLS fit", b.R2, 0.94)
    number(w, "m of the linearised OLS fit", b.m_OLS, 1.55)
    number(w, "R^2 of the OLS fit in share space", b.R2_OLS, 0.48)
    number(w, "exp_lb, mean social assistance norm 2021-2023 (EUR)", b.exp_lb, 1087, 0)
    number(w, "mean unemployment rate 2021-2025 (%)", b.eta_data_pct, 3.77)
    return b


# ─── Calibration (main text and SI) ─────────────────────────────────────────────────────

def calibration(baseline):
    s, pos, accepted, band, ks = abc_results()
    w = "SI Calibration"
    acc5, acc1 = accepted[0.05], accepted[0.01]
    main = ks[ks.tol == 0.05].set_index("param")
    strict = ks[ks.tol == 0.01].set_index("param")
    boot = abc_bootstrap(pos, acc5, main.ks_p95.iloc[0]).set_index("param")

    number("Fig. 2 caption", "parameter combinations", len(s), 1001, 0)
    number(w, "retained at the 5% tolerance", len(acc5), 51, 0)
    number(w, "retained at the 1% tolerance", len(acc1), 11, 0)
    runs = read("abc_calibration/runs_set0.csv.gz")
    se = runs.groupby("sample_id").rate_tot.agg(lambda x: x.std() / np.sqrt(len(x))).reindex(acc5)
    number(w, "median s.e. of the seed-mean foreclosure rate, retained (pp)", se.median(), 0.008, 3)
    number(w, "max-T threshold, 5% tolerance", main.ks_p95.iloc[0], 0.23)
    number(w, "max-T threshold, 1% tolerance", strict.ks_p95.iloc[0], 0.49)
    for p, v, lo, hi, q05, q95, scale, dec in [("eta", 0.71, 0.68, 0.75, 0.032, 0.056, 1, 3),
                                               ("m", 0.55, 0.51, 0.61, 4.38, 5.90, 1, 2),
                                               ("varepsilon", 0.33, 0.28, 0.43, 1.78, 2.55, 1e3, 2)]:
        number(w, f"KS of {p}, 5%", main.loc[p, "ks"], v)
        number(w, f"bootstrap 90% interval of the KS of {p}, lower", boot.loc[p, "ks_lo"], lo)
        number(w, f"bootstrap 90% interval of the KS of {p}, upper", boot.loc[p, "ks_hi"], hi)
        vals = s.loc[acc5, p] * scale
        number(w, f"retained 5th percentile of {p}" + (" (x1e3)" if scale != 1 else ""), vals.quantile(0.05), q05, dec)
        number(w, f"retained 95th percentile of {p}" + (" (x1e3)" if scale != 1 else ""), vals.quantile(0.95), q95, dec)
        number(w, f"share of resamples with the KS of {p} above the threshold", boot.loc[p, "p_above"], 1.0)
    bounds = load_problem()
    number(w, "range of eta, lower", bounds["eta"][0], 0.03)
    number(w, "range of eta, upper", bounds["eta"][1], 0.13)
    number(w, "range of m, lower", bounds["m"][0], 4.34)
    number(w, "range of m, upper", bounds["m"][1], 8.06)
    number(w, "range of varepsilon, lower (x1e3)", 1e3 * bounds["varepsilon"][0], 1.39)
    number(w, "range of varepsilon, upper (x1e3)", 1e3 * bounds["varepsilon"][1], 2.57)

    # KS on the two halves of the Sobol' sequence, tolerance fixed
    _, d_tol = abc_distance(s)
    diffs = []
    for half in (s.raw_index < 2 ** 12, s.raw_index >= 2 ** 12):
        sk = s[half]
        ak = sk.index[sk.abc_distance <= d_tol[0.05]]
        diffs.append([stats.ks_2samp(pos.loc[ak, p], pos.loc[sk.index, p]).statistic for p in ("eta", "m", "varepsilon")])
    number(w, "largest difference of the KS of eta, m, varepsilon between the two halves (< 0.05)",
           float(np.max(np.abs(np.diff(diffs, axis=0)))) < 0.05, 1, 0)
    for p, v in [("eta", 0.80), ("m", 0.56), ("varepsilon", 0.42)]:
        number(w, f"KS of {p}, 1% tolerance", strict.loc[p, "ks"], v)
    number(w, "varepsilon below the threshold at 1% (1 = yes)", float(not strict.loc["varepsilon", "constrained"]), 1, 0)
    others = [p for p in ABC_PARAMS if p not in ("eta", "m", "varepsilon")]
    number(w, "KS of the other six parameters, smallest", main.loc[others, "ks"].min(), 0.10)
    number(w, "KS of the other six parameters, largest", main.loc[others, "ks"].max(), 0.14)
    number(w, "share of resamples below the threshold, other six, smallest (at least)",
           1 - boot.loc[others, "p_above"].max(), 0.83)
    number(w, "Spearman correlation of m and varepsilon among the retained",
           stats.spearmanr(s.loc[acc5, "m"], s.loc[acc5, "varepsilon"]).statistic, 0.54)

    number(w, "correlation of the log outcomes across combinations",
           np.corrcoef(np.log(s.rate_tot), np.log(s.arrears_fin))[0, 1], 0.93)
    (flo, fhi), (dlo, dhi) = EMPIRICAL_BANDS["rate_tot"], EMPIRICAL_BANDS["arrears_fin"]
    in_bkr = s.arrears_fin.between(dlo, dhi)
    in_nhg = s.rate_tot.between(flo, fhi)
    number(w, "combinations with distress inside the BKR band", in_bkr.sum(), 512, 0)
    number(w, "combinations with the rate inside the NHG band", in_nhg.sum(), 6, 0)
    number(w, "combinations inside both bands", (in_bkr & in_nhg).sum(), 1, 0)
    number(w, "retained foreclosure rate, 5th percentile (%)", s.loc[acc5, "rate_tot"].quantile(0.05), 0.43)
    number(w, "retained foreclosure rate, 95th percentile (%)", s.loc[acc5, "rate_tot"].quantile(0.95), 0.65)
    number(w, "retained distress share, 5th percentile (%)", s.loc[acc5, "arrears_fin"].quantile(0.05), 0.56)
    number(w, "retained distress share, 95th percentile (%)", s.loc[acc5, "arrears_fin"].quantile(0.95), 1.49)

    # joint 95% confidence region of (m, varepsilon) of the baseline fit (units m, varepsilon x 1e3)
    est = np.array([baseline.m, 1e3 * baseline.varepsilon])
    se = np.array([baseline.se_m, 1e3 * baseline.se_varepsilon])
    cov = np.outer(se, se) * np.array([[1, baseline.corr_m_varepsilon], [baseline.corr_m_varepsilon, 1]])
    v = np.column_stack([s.m, 1e3 * s.varepsilon]) - est
    inside = np.einsum("ij,jk,ik->i", v, np.linalg.inv(cov), v) <= 5.99
    number(w, "retained combinations inside the joint 95% region", inside[s.index.isin(acc5)].sum(), 0, 0)
    number(w, "lowest foreclosure rate inside the region (%)", s.rate_tot[inside].min(), 0.66)
    number(w, "lowest distress share inside the region (%)", s.arrears_fin[inside].min(), 1.56)
    lo_m, hi_m = bounds["m"]
    number(w, "baseline m above the retained range (1 = yes)", float(6.2 > s.loc[acc5, "m"].quantile(0.95)), 1, 0)
    number(w, "bootstrap 90% interval of the retained 95th percentile of m, lower",
           lo_m + boot.loc["m", "q95_lo"] * (hi_m - lo_m), 5.69)
    number(w, "bootstrap 90% interval of the retained 95th percentile of m, upper",
           lo_m + boot.loc["m", "q95_hi"] * (hi_m - lo_m), 6.11)
    for p, base in [("eta", 0.037), ("varepsilon", 0.00198)]:
        q = s.loc[acc5, p].quantile([0.05, 0.95])
        number(w, f"baseline {p} inside the retained range (1 = yes)", float(q.iloc[0] <= base <= q.iloc[1]), 1, 0)

    star = band[0]
    number("SI Table 1", "calibrated eta", s.loc[star, "eta"], 0.0383, 4)
    number("SI Table 1", "calibrated m", s.loc[star, "m"], 5.25)
    number("Fig. 5 caption", "calibrated varepsilon", s.loc[star, "varepsilon"], 0.00250, 5)
    for k, v in CALIBRATED.items():
        assert np.isclose(s.loc[star, k], v), k

    c = read("abc_calibration/calibrated/runs_set0.csv.gz")
    number("Main text, calibration", "foreclosure rate at S0, calibrated, beta1 = 1, beta2 = 0.5 (%)", c.rate_tot.mean(), 0.42)
    number("Main text, calibration", "share of households in financial distress, same (%)", c.arrears_fin.mean(), 0.51)


# ─── Results: foreclosure rates and composition (Fig. 3, SI Fig. 2) ──────────────────────

def rates():
    r = load_cascade_runs()
    sl = r[(r.kappa == 10) & (r.vartheta == 0.05) & (r.beta2 == 0.5)].copy()
    for ch in ("tot", "fin", "strat"):
        sl[f"rate_{ch}"] = 100 * sl[f"n_fcl_{ch}"] / sl.mean_active / YEARS
    z = sl[sl.delta == 0]
    m = z.groupby("setting")[["rate_tot", "rate_fin", "rate_strat"]].mean()
    w = "Main text, Fig. 3a"
    number(w, "runs at S0", (z.setting == 0).sum(), 20, 0)
    number(w, "runs per flooded scenario", z[z.setting > 0].groupby("setting").size().min(), 60, 0)
    number(w, "annual foreclosure rate S0 (%)", m.rate_tot[0], 0.42)
    number(w, "annual foreclosure rate S4 (%)", m.rate_tot[4], 0.91)
    number(w, "financial rate S0 (%)", m.rate_fin[0], 0.42)
    number(w, "financial rate, lowest of S1-S4 (%)", m.rate_fin[1:].min(), 0.36)
    number(w, "financial rate, highest of S1-S4 (%)", m.rate_fin[1:].max(), 0.36)
    number(w, "strategic rate S0 (%)", m.rate_strat[0], 0)
    number(w, "strategic rate S4 (%)", m.rate_strat[4], 0.55)
    inc = m.rate_tot - m.rate_tot[0]
    number(w, "share of the S0-S4 increase reached at S1", inc[1] / inc[4], 0.50)

    # seed as unit (Statistics section): lambda averaged within seed
    for s, printed in [(0, 0.42), (4, 0.91)]:
        number("Statistics, seed intervals", f"annual rate S{s}, mean over seeds (%)",
               t_mean(z[z.setting == s].groupby("seed").rate_tot.mean())[0], printed)

    # strategic foreclosures in homes undamaged in the month of foreclosure (delta = 0)
    cells = read("cascades_rnet/fcl_cells.csv.gz").merge(r[["run_id", "setting"]], on="run_id")
    st = cells[(cells.channel == 0) & (cells.setting > 0)]
    s4 = st[st.setting == 4]
    number("Main text, introduction", "strategic foreclosures in undamaged homes, S4 (%)",
           100 * s4.n_fcl[s4.flooded == 0].sum() / s4.n_fcl.sum(), 37.6, 1)
    number("Discussion", "strategic foreclosures in undamaged homes, S1-S4 (%)",
           100 * st.n_fcl[st.flooded == 0].sum() / st.n_fcl.sum(), 39, 0)

    # composition of the sensitivity-analysis vectors (Fig. 3b)
    design = load_saltelli_matrix()[["saltelli_sample_id", "beta1", "beta2", "delta"]]
    v = load_gsa_vectors().merge(design, on="saltelli_sample_id")
    v["gap"] = v.beta1 - v.beta2
    small = v[v.delta <= 0.15]
    w = "Main text, Fig. 3b"
    number(w, "vectors with delta <= 0.15", small.saltelli_sample_id.nunique(), 786, 0)
    number(w, "of which beta1 - beta2 >= 1/2", small[small.gap >= 0.5].saltelli_sample_id.nunique(), 291, 0)
    wide = small[small.gap >= 0.5]
    share = wide.groupby("setting").n_strategic.sum() / wide.groupby("setting").n_total.sum()
    number(w, "strategic share S0, beta1 - beta2 >= 1/2, delta <= 0.15 (%)", 100 * share[0], 20, 0)
    number(w, "strategic share S4, same (%)", 100 * share[4], 28, 0)


# ─── Results: foreclosure timing and lender exposure (Fig. 5) ────────────────────────────

def timing():
    fcl_gap, fcl_b3 = load_timing("gap"), load_timing("beta3")
    fcl = fcl_gap[fcl_gap.gap == 0.5].reset_index(drop=True)
    flooded = fcl[fcl.setting > 0]
    fin, strat = flooded[flooded.default_reason == 1], flooded[flooded.default_reason == -1]
    strat_share = lambda d: 100 * (d.default_reason == -1).mean()
    w = "Main text, Fig. 5"
    number("Fig. 5 caption", "foreclosures", len(fcl), 68927, 0)
    number(w, "median age S0 (months)", fcl[fcl.setting == 0].t.median(), 62, 0)
    number(w, "median age S4 (months)", fcl[fcl.setting == 4].t.median(), 37, 0)
    number(w, "median age, strategic, S1-S4 (months)", strat.t.median(), 15, 0)
    number(w, "median age, financial, S1-S4 (months)", fin.t.median(), 63, 0)
    by_ch = fcl.groupby(["default_reason", "setting"]).t.median()
    number(w, "financial median age, lowest over scenarios", by_ch.loc[1].min(), 62, 0)
    number(w, "financial median age, highest over scenarios", by_ch.loc[1].max(), 64, 0)
    number(w, "strategic median age, lowest over S1-S4", by_ch.loc[-1].min(), 14, 0)
    number(w, "strategic median age, highest over S1-S4", by_ch.loc[-1].max(), 15, 0)
    number(w, "strategic share S0 (%)", strat_share(fcl[fcl.setting == 0]), 0, 0)
    number(w, "strategic share S4 (%)", strat_share(fcl[fcl.setting == 4]), 60, 0)

    # outstanding balance at foreclosure, % of the original loan: (121 - t) / 120 for mortgages
    # originated during the run (the t = 0 stock has an unknown original term)
    bal = fcl[(fcl.t - fcl.step) < 0].assign(bal=lambda d: 100 * (121 - d.t) / 120)
    w = "Main text, lender exposure"
    number(w, "balance at foreclosure S0 (% of the loan)", bal[bal.setting == 0].bal.mean(), 48, 0)
    number(w, "balance at foreclosure S4 (% of the loan)", bal[bal.setting == 4].bal.mean(), 68, 0)
    bfl = bal[bal.setting > 0]
    number(w, "balance at strategic foreclosure, S1-S4 (% of the loan)", bfl[bfl.default_reason == -1].bal.mean(), 83, 0)

    # seed as unit: S0 runs coincide across lambda, keep one copy
    s0 = fcl[fcl.setting == 0]
    distinct = pd.concat([fcl[fcl.setting > 0], s0[s0["lambda"] == s0["lambda"].min()]])
    w = "Statistics, seed intervals"
    for s, printed in [(0, 62), (4, 37)]:
        number(w, f"median age S{s}, mean of per-seed medians", t_mean(distinct[distinct.setting == s].groupby("seed").t.median())[0], printed, 0)
    fl = distinct[distinct.setting > 0]
    for reason, name, printed in [(-1, "strategic", 15), (1, "financial", 63)]:
        number(w, f"median age {name}, S1-S4, mean of per-seed medians",
               t_mean(fl[fl.default_reason == reason].groupby("seed").t.median())[0], printed, 0)

    # SI, foreclosure timing
    w = "SI Foreclosure timing"
    number(w, "financial: median distress months, S1-S4", fin.n_dis.median(), 27, 0)
    number(w, "financial: share of distress months that are strategic triggers (%)",
           100 * fin.n_dis_strat.sum() / fin.n_dis.sum(), 92, 0)
    number(w, "strategic: median distress months, S1-S4", strat.n_dis.median(), 6, 0)
    med_n = fcl.groupby("setting").n_dis.median()
    number(w, "pooled median distress months S0", med_n[0], 27, 0)
    number(w, "pooled median distress months S4", med_n[4], 13, 0)
    by_n = fcl.groupby(["default_reason", "setting"]).n_dis.median()
    number(w, "financial median distress months, lowest over scenarios", by_n.loc[1].min(), 27, 0)
    number(w, "financial median distress months, highest over scenarios", by_n.loc[1].max(), 28, 0)
    number(w, "strategic median distress months, lowest over S1-S4", by_n.loc[-1].min(), 6, 0)
    number(w, "strategic median distress months, highest over S1-S4", by_n.loc[-1].max(), 7, 0)
    s4 = fcl[fcl.setting == 4]
    for lam, printed in [(0.25, 34), (0.5, 61), (0.75, 72)]:
        number(w, f"strategic share S4, lambda = {lam} (%)", strat_share(s4[s4["lambda"] == lam]), printed, 0)
    fin4 = s4[s4.default_reason == 1].groupby("lambda").t.median()
    number(w, "financial median age S4, lowest over lambda", fin4.min(), 63, 0)
    number(w, "financial median age S4, highest over lambda", fin4.max(), 64, 0)
    for lam, printed in [(0.25, 21), (0.75, 13)]:
        number(w, f"strategic median age S4, lambda = {lam}", s4[(s4.default_reason == -1) & (s4["lambda"] == lam)].t.median(), printed, 0)
    g = fcl_gap[fcl_gap.setting > 0]
    number(w, "strategic share S1-S4, gap 0.7 (%)", strat_share(g[g.gap == 0.7]), 21, 0)
    number(w, "strategic share S1-S4, gap 0.45 (%)", strat_share(g[g.gap == 0.45]), 96, 0)
    med_gap = g.groupby("gap").n_dis.median()
    number(w, "pooled median distress months, gap >= 0.5, lowest", med_gap[med_gap.index >= 0.5].min(), 16, 0)
    number(w, "pooled median distress months, gap >= 0.5, highest", med_gap[med_gap.index >= 0.5].max(), 19, 0)
    number(w, "pooled median distress months, gap 0.45", med_gap[0.45], 7, 0)
    g0 = fcl_gap[(fcl_gap.setting == 0) & (fcl_gap.gap == 0.45)]
    number(w, "strategic share S0, gap 0.45 (%)", strat_share(g0), 91, 0)
    gfin = g[g.default_reason == 1].groupby("gap").n_dis.median()
    number(w, "financial median distress months, gap 0.7", gfin[0.7], 19, 0)
    number(w, "financial median distress months, gap 0.5", gfin[0.5], 27, 0)
    b3 = pd.concat([flooded, fcl_b3[fcl_b3.setting > 0]])
    number(w, "strategic share S1-S4, beta3 = 2 (%)", strat_share(b3[b3.beta3 == 2.0]), 45, 0)
    number(w, "strategic share S1-S4, beta3 = 0.5 (%)", strat_share(b3[b3.beta3 == 0.5]), 62, 0)


def main():
    baseline = initialization()
    calibration(baseline)
    rates()
    timing()

    tab = pd.DataFrame(ROWS)
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    tab.to_csv(os.path.join(ROOT, "output", "numbers.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 1000, "display.max_colwidth", 80):
        print(tab.assign(computed=tab.computed.round(4)).to_string(index=False))
    checked = tab[tab.printed.notna()]
    bad = checked[~checked.match.astype(bool)]
    print(f"\n{len(checked) - len(bad)} of {len(checked)} numbers match the paper")
    if len(bad):
        print(bad.to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
