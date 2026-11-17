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

from utils import (ABC_PARAMS, CALIBRATED, EMPIRICAL_BANDS, GEV_SCENARIOS, PREPARATION, ROOT, TIERS, YEARS, abc_bootstrap, abc_distance,
                   abc_results, compare_methods, draw_applicants, gsa_outcome, gsa_problem, load_attribution_free, load_cascade_runs,
                   load_cascade_sizes, load_gsa_runs, load_gsa_vectors, load_paired_seed, load_problem,
                   load_saltelli_matrix, load_timing, map_fields, pawn_design, pawn_indices, pawn_threshold,
                   r_by_scenario, read, read_map, size_density, sobol_indices, sparks_and_sweeps, summarise)

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


# ─── SI, Flood scenarios and flood maps ──────────────────────────────────────────────────

def gev_peak(mu, sigma, xi):
    """Mode and density at the mode of GEV(mu, sigma, xi) (SI, equation of the GEV peak)."""
    return mu + sigma / xi * ((1 + xi) ** (-xi) - 1), (1 + xi) ** (1 + xi) * np.exp(-(1 + xi)) / sigma


def flood_scenarios_and_maps():
    w = "SI Flood scenarios"
    x0, f0 = gev_peak(10.0, 27.0, 0.2)
    number(w, "mode of the reference GEV(10, 27, 0.2)", x0, 5.17)
    number(w, "density at the mode of the reference", f0, 0.0139, 4)
    peaks = np.array([gev_peak(*p) for p in GEV_SCENARIOS.values()])
    number(w, "mode of the scenarios, lowest", peaks[:, 0].min(), 5.0, 1)
    number(w, "mode of the scenarios, highest", peaks[:, 0].max(), 7.4, 1)
    number(w, "density at the mode of the scenarios, lowest", peaks[:, 1].min(), 0.0134, 4)
    number(w, "density at the mode of the scenarios, highest", peaks[:, 1].max(), 0.0143, 4)

    w = "SI Fig. 2 caption"
    region = read_map("nvm_cleaned.png") > 0                     # cells with housing
    for (tier, f), printed in zip(TIERS, (4.4, 9.1, 12.1, 14.3)):
        number(w, f"flooded share of the region, {tier} map (%)", 100 * ((read_map(f) > 0) & region).sum() / region.sum(), printed, 1)
    housing, region_fig, band, rate0, rate4, ratio = map_fields(4, 0.05)
    number(w, "local rate S4 / S0, median over cells inside the flood extent", 2 ** np.nanmedian(ratio[band > 0]), 5.2, 1)
    number(w, "local rate S4 / S0, median over cells outside the flood extent", 2 ** np.nanmedian(ratio[band == 0]), 1.4, 1)
    number(w, "masked cells, smoothed housing density below 5% of its maximum (% of the region)",
           100 * (region_fig & np.isnan(rate0)).sum() / region_fig.sum(), 2.1, 1)


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


# SI Table 5 as printed: attribute -> (applicants, admitted), each (median, lower quartile, upper quartile)
TABLE_S5 = {"income": ((2081, 1225, 3198), (3523, 2868, 4619), 0), "zeta": ((0.65, 0.38, 0.90), (0.92, 0.80, 1.06), 2),
            "install_share": ((0.43, 0.24, 0.84), (0.19, 0.13, 0.27), 2), "install": ((878, 608, 1255), (687, 503, 901), 0),
            "ltv": ((0.57, 0.50, 0.64), (0.54, 0.47, 0.59), 2), "v": ((187, 133, 261), (155, 115, 202), 0),
            "sen": ((21, 9, 39), (21, 9, 39), 0)}


def admission():
    """Applicants and admitted households at entry (utils.draw_applicants, 2 x 10^6 draws, seed 0)."""
    a = draw_applicants()
    adm = a[a.admitted]
    w = "SI Agent initialisation (admission)"
    number(w, "applicants admitted (%)", 100 * a.admitted.mean(), 16.5, 1)
    number(w, "median funds over loan of the applicants", a.zeta.median(), 0.65)
    number(w, "admitted with income above 3000 EUR per month (%)", 100 * (adm.income > 3000).mean(), 70, 0)
    number(w, "admitted with funds above 80% of the loan (%)", 100 * (adm.zeta > 0.8).mean(), 75, 0)
    number(w, "applicants with income above 3000 EUR per month (%)", 100 * (a.income > 3000).mean(), 28, 0)
    number(w, "applicants with funds above 80% of the loan (%)", 100 * (a.zeta > 0.8).mean(), 35, 0)
    number(w, "instalment over income, median admitted (%)", 100 * adm.install_share.median(), 19, 0)
    number(w, "instalment over income, median applicant (%)", 100 * a.install_share.median(), 43, 0)
    number(w, "admitted paying less than 40% of income (%)", 100 * (adm.install_share < 0.4).mean(), 95, 0)
    number(w, "applicants paying less than 40% of income (%)", 100 * (a.install_share < 0.4).mean(), 46, 0)
    number(w, "applicants in the first seniority band (%)", 100 * (a.sen <= 100).mean(), 99.95)
    w = "SI Table 5"
    for col, (app, admd, d) in TABLE_S5.items():
        for who, x, printed in (("applicants", a[col], app), ("admitted", adm[col], admd)):
            for stat, q, p in zip(("median", "lower quartile", "upper quartile"), (50, 25, 75), printed):
                number(w, f"{col}, {who}, {stat}", np.percentile(x, q), p, d)


def zeta_scale():
    """Test of the scale of initial liquid funds (experiments/zeta_scale/): seeds averaged per scale."""
    z = read(os.path.join(ROOT, "data", "zeta_scale", "runs.csv.gz")).groupby("zeta_scale").mean(numeric_only=True)
    w = "SI Agent initialisation (zeta_scale)"
    number(w, "scale values", len(z), 101, 0)
    mid = z[(z.index >= 0.008 - 1e-9) & (z.index <= 0.02 + 1e-9)]
    number(w, "foreclosure rate, scales 0.008-0.02, lowest (%)", mid.rate_tot.min(), 0.39)
    number(w, "foreclosure rate, scales 0.008-0.02, highest (%)", mid.rate_tot.max(), 0.46)
    number(w, "financial distress, scales 0.008-0.02, lowest (%)", mid.arrears_fin.min(), 0.46)
    number(w, "financial distress, scales 0.008-0.02, highest (%)", mid.arrears_fin.max(), 0.54)
    for s, age, acc, pop, dec in ((0.01, 51, 5.9, 960, -1), (0.015, 61, 16.5, 2700, -2), (0.02, 69, 22.6, 3700, -2)):
        r = z.loc[np.isclose(z.index, s)].iloc[0]
        number(w, f"median loan age at financial foreclosure, scale {s}, mean over seeds (months)", r.median_age_fin, age, 0)
        number(w, f"applicants admitted, scale {s} (%)", 100 * r.acceptance, acc, 1)
        number(w, f"mean active households, scale {s} (about)", r.mean_active, pop, dec)


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


# ─── Results: spatial pattern (SI Fig. 2) and SI population and composition ──────────────

def maps_and_composition():
    housing, region, band, rate0, rate4, ratio = map_fields(4, 0.05)
    w = "Main text, SI Fig. 2"
    number(w, "local rate S4 / S0 inside the flood extent, median (about fivefold)", 2 ** np.nanmedian(ratio[band > 0]), 5, 0)
    number(w, "local rate S4 / S0 outside the flood extent, median (less than 1.5)", float(2 ** np.nanmedian(ratio[band == 0]) < 1.5), 1, 0)

    runs = load_gsa_runs()
    by_set = runs.groupby(["saltelli_sample_id", "setting"]).mean_active.mean().unstack()
    par = runs.groupby("saltelli_sample_id")[["beta1", "beta2", "delta"]].first()
    par["gap"], par["pop0"] = par.beta1 - par.beta2, by_set[0]
    w = "SI Fig. 3, population"
    dr = par[par.gap >= 0.5]
    number(w, "vectors with beta1 - beta2 >= 1/2", len(dr), 1162, 0)
    number(w, "their smallest population at S0", dr.pop0.min(), 514, 0)
    number(w, "their largest population at S0", dr.pop0.max(), 2709, 0)
    number(w, "median population, gap >= 1/2, delta <= 0.15", dr[dr.delta <= 0.15].pop0.median(), 2550, 0)
    number(w, "median population, gap >= 1/2, delta >= 0.35", dr[dr.delta >= 0.35].pop0.median(), 1566, 0)
    near = par[(par.gap > 0.4) & (par.gap < 0.5)]
    number(w, "median population, 0.4 < gap < 0.5", near.pop0.median(), 1229, 0)
    number(w, "median population, gap <= 0.1", par[par.gap <= 0.1].pop0.median(), 16, 0)
    number(w, "median population, 0.4 < gap < 0.5, delta <= 0.15", near[near.delta <= 0.15].pop0.median(), 2341, 0)
    number(w, "median population, 0.4 < gap < 0.5, delta >= 0.35", near[near.delta >= 0.35].pop0.median(), 622, 0)

    rate = runs.groupby(["saltelli_sample_id", "setting"]).rate_tot.mean().unstack()
    number("SI Default composition", "vectors with a higher rate at S4 than at S0", (rate[4] > rate[0]).sum(), 3286, 0)
    number("SI Default composition", "vectors in the design", len(rate), 3328, 0)

    v = load_gsa_vectors().merge(load_saltelli_matrix()[["saltelli_sample_id", "beta1", "beta2", "delta"]], on="saltelli_sample_id")
    v["gap"] = v.beta1 - v.beta2
    small = v[v.delta <= 0.15].copy()
    small["bin"] = pd.qcut(small.gap, 12)
    share = lambda g: 100 * g.n_strategic.sum() / g.n_total.sum()
    strat = small.groupby(["bin", "setting"], observed=True)[["n_strategic", "n_total"]].apply(share).unstack()
    active = runs.groupby(["setting", "saltelli_sample_id"], as_index=False).mean_active.mean()
    act = small[["setting", "saltelli_sample_id", "bin"]].merge(active).groupby(["bin", "setting"], observed=True).mean_active.mean().unstack()
    narrow = [b for b in strat.index if b.right <= 0.46]
    w = "SI Default composition"
    number(w, "strategic share in the bins below 0.46, lowest over scenarios (above 92%)", float(strat.loc[narrow].min().min() > 92), 1, 0)
    number(w, "active mortgages in the bins below 0.46, largest (below 1,600)", float(act.loc[narrow].max().max() < 1600), 1, 0)
    widest = strat.index[-1]
    number(w, "strategic share in the widest bin, lowest (%)", strat.loc[widest].min(), 1.5, 1)
    number(w, "strategic share in the widest bin, highest (%)", strat.loc[widest].max(), 3, 0)
    number(w, "active mortgages in the widest bin (about 2,500)", round(act.loc[widest].mean(), -2), 2500, 0)
    number(w, "largest rise of the strategic share S4 - S0 within a bin (pp)", (strat[4] - strat[0]).max(), 13, 0)
    r = v[(v.gap >= 0.5) & (v.setting == 0)]
    number(w, "strategic share S0, gap >= 1/2, 0.15 < delta <= 0.35 (%)", share(r[(r.delta > 0.15) & (r.delta <= 0.35)]), 63, 0)
    number(w, "strategic share S0, gap >= 1/2, delta > 0.35 (%)", share(r[r.delta > 0.35]), 83, 0)


# ─── SI, Global sensitivity analysis ─────────────────────────────────────────────────────

def sensitivity():
    names = gsa_problem()["names"]
    sob, paw, thr = {}, {}, {}
    for Y in ("perc_fin", "n_financial", "n_strategic", "mean_months"):
        si = sobol_indices(Y)
        sob[Y] = pd.concat([pd.DataFrame({"param": names, "setting": s, "S1": d["S1"], "S1_conf": d["S1_conf"],
                                          "ST": d["ST"], "ST_conf": d["ST_conf"]}) for s, d in si.items()])
        pw = pawn_indices(Y)
        paw[Y] = pd.concat([pd.DataFrame({"param": names, "setting": s, "T": np.asarray(d["median"], float)})
                            for s, d in pw.items()])
        thr[Y] = pawn_threshold(Y)
        paw[Y]["sig"] = paw[Y]["T"] > thr[Y]

    def rng(table, Y, p, col):
        x = table[Y][table[Y].param == p][col]
        return x.min(), x.max()

    def both(where, what, lo_hi, lo, hi, dec=2):
        number(where, what + ", lowest", lo_hi[0], lo, dec)
        number(where, what + ", highest", lo_hi[1], hi, dec)

    w = "SI Sensitivity analysis"
    X = pawn_design()
    number(w, "correlation of beta1 and beta2 on the A and B blocks", np.corrcoef(X[:, 1], X[:, 2])[0, 1], 0.36)
    sub = (X[:, 1] > 0.5) & (X[:, 2] < 0.5)
    number(w, "sub-region beta1 > 0.5, beta2 < 0.5: design points", sub.sum(), 322, 0)
    number(w, "sub-region: design points of the A and B blocks", len(X), 512, 0)
    number(w, "sub-region: correlation of beta1 and beta2", np.corrcoef(X[sub, 1], X[sub, 2])[0, 1], -0.02)
    from SALib.analyze import pawn as salib_pawn
    problem = gsa_problem()
    subT = {}
    for Y in ("perc_fin", "n_financial", "n_strategic"):
        subT[Y] = pd.concat([pd.DataFrame({"param": names, "setting": s, "T": np.asarray(salib_pawn.analyze(
            problem, X[sub], gsa_outcome(s, Y)[:len(X)][sub], S=10, seed=0)["median"], float)}) for s in range(5)])
    allT = pd.concat(subT.values(), keys=subT.keys(), names=["Y"]).reset_index()
    gap = allT.pivot_table(index=["Y", "setting"], columns="param", values="T")
    both(w, "sub-region: T_beta1 - T_beta2 over outcomes and scenarios", ((gap.beta1 - gap.beta2).min(), (gap.beta1 - gap.beta2).max()), 0.09, 0.13)
    both(w, "sub-region: T_beta1, share of financial defaults", rng(subT, "perc_fin", "beta1", "T"), 0.32, 0.33)
    both(w, "sub-region: T_beta2, share of financial defaults", rng(subT, "perc_fin", "beta2", "T"), 0.22, 0.23)
    both(w, "sub-region: T_beta2, all outcomes", (gap.beta2.min(), gap.beta2.max()), 0.20, 0.23)
    both(w, "sub-region: T_kappa, all outcomes", (gap.kappa.min(), gap.kappa.max()), 0.21, 0.23)
    both(w, "sub-region: T_delta, all outcomes", (gap.delta.min(), gap.delta.max()), 0.18, 0.26)
    gm = gap.loc[["perc_fin", "n_financial"]].m
    both(w, "sub-region: T_m, share of financial defaults and n_financial", (gm.min(), gm.max()), 0.20, 0.27)

    w = "SI GSA results"
    for Y, n in [("perc_fin", 19), ("n_financial", 10), ("n_strategic", 16)]:
        number(w, f"significant PAWN comparisons of 55, {Y}", paw[Y].sig.sum(), n, 0)
    below = pd.concat([paw[Y][~paw[Y].sig] for Y in ("perc_fin", "n_financial", "n_strategic")])
    both(w, "T of the comparisons below the threshold", (below["T"].min(), below["T"].max()), 0.07, 0.15)

    pf, nf, ns, mm = "perc_fin", "n_financial", "n_strategic", "mean_months"
    both(w, "S1 of beta1, share of financial defaults", rng(sob, pf, "beta1", "S1"), 0.35, 0.37)
    both(w, "S1 of beta2, share of financial defaults", rng(sob, pf, "beta2", "S1"), 0.24, 0.26)
    clear = sob[pf].assign(c=sob[pf].S1 - sob[pf].S1_conf > 0).groupby("param").c.sum()
    number(w, "parameters whose S1 interval clears zero at every scenario, share of financial defaults (beta1, beta2)",
           float(set(clear[clear == 5].index) == {"beta1", "beta2"}), 1, 0)
    s1sum = sob[pf].groupby("setting").S1.sum()
    both(w, "sum of the first-order indices, share of financial defaults", (s1sum.min(), s1sum.max()), 0.73, 0.78)
    both(w, "ST of beta2, share of financial defaults", rng(sob, pf, "beta2", "ST"), 0.44, 0.46)
    both(w, "S1 of kappa, share of financial defaults", rng(sob, pf, "kappa", "S1"), 0.04, 0.05)
    k = sob[pf][sob[pf].param == "kappa"]
    number(w, "S1 interval of kappa includes zero at every scenario (1 = yes)", float(((k.S1 - k.S1_conf) <= 0).all()), 1, 0)
    both(w, "ST of kappa, share of financial defaults", rng(sob, pf, "kappa", "ST"), 0.19, 0.20)
    number(w, "S1 and ST of beta3 and varepsilon, share of financial defaults, all below about 0.03 (1 = yes)",
           float(sob[pf][sob[pf].param.isin(["beta3", "varepsilon"])][["S1", "ST"]].max().max() <= 0.03), 1, 0)
    number(w, "scenarios with T_kappa above the threshold, share of financial defaults", paw[pf][paw[pf].param == "kappa"].sig.sum(), 3, 0)
    number(w, "scenarios with T_beta3 above the threshold, share of financial defaults", paw[pf][paw[pf].param == "beta3"].sig.sum(), 1, 0)
    both(w, "S1 of beta1, n_strategic", rng(sob, ns, "beta1", "S1"), 0.47, 0.48)
    both(w, "S1 of beta1, n_financial", rng(sob, nf, "beta1", "S1"), 0.24, 0.26)
    both(w, "ST of beta1, n_financial", rng(sob, nf, "beta1", "ST"), 0.54, 0.55)
    both(w, "ST of beta2, n_financial", rng(sob, nf, "beta2", "ST"), 0.26, 0.30)
    both(w, "ST of beta2, n_strategic", rng(sob, ns, "beta2", "ST"), 0.29, 0.31)
    number(w, "ST of beta3, both counts, largest", max(rng(sob, nf, "beta3", "ST")[1], rng(sob, ns, "beta3", "ST")[1]), 0.06)
    leads = all((t.groupby("setting").apply(lambda g: g.set_index("param")[c].idxmax(), include_groups=False) == "beta1").all()
                for Y in (pf, nf, ns) for t, c in ((sob[Y], "ST"), (paw[Y], "T")))
    number(w, "beta1 has the largest ST and T at every scenario, all three outcomes (1 = yes)", float(leads), 1, 0)
    both(w, "ST of beta1, share of financial defaults", rng(sob, pf, "beta1", "ST"), 0.58, 0.60)
    both(w, "T of beta1, share of financial defaults", rng(paw, pf, "beta1", "T"), 0.29, 0.30)
    both(w, "ST of beta1, n_strategic", rng(sob, ns, "beta1", "ST"), 0.59, 0.60)
    both(w, "T of beta1, n_strategic", rng(paw, ns, "beta1", "T"), 0.29, 0.29)
    b2 = pd.concat([sob[Y][sob[Y].param == "beta2"] for Y in (pf, nf, ns)])
    both(w, "S1 of beta2, three outcomes", (b2.S1.min(), b2.S1.max()), 0.19, 0.26)
    both(w, "ST of beta2, three outcomes", (b2.ST.min(), b2.ST.max()), 0.26, 0.46)
    pb2 = pd.concat([paw[Y][paw[Y].param == "beta2"] for Y in (pf, nf, ns)])
    both(w, "T of beta2, three outcomes", (pb2["T"].min(), pb2["T"].max()), 0.14, 0.15)
    number(w, "T of beta2 below the threshold everywhere (1 = yes)", float(not pb2.sig.any()), 1, 0)
    both(w, "ST of delta, share of financial defaults", rng(sob, pf, "delta", "ST"), 0.26, 0.27)
    d1 = pd.concat([sob[Y][sob[Y].param == "delta"] for Y in (pf, ns)])
    both(w, "S1 of delta, share of financial defaults and n_strategic", (d1.S1.min(), d1.S1.max()), 0.08, 0.09)
    for Y, t in [(pf, 0.17), (ns, 0.19)]:
        number(w, f"T_delta above the threshold at every scenario, {Y} (1 = yes)", float(paw[Y][paw[Y].param == "delta"].sig.all()), 1, 0)
        both(w, f"T of delta, {Y}", rng(paw, Y, "delta", "T"), t, t)
    lam = pd.concat([sob[Y][sob[Y].param == "lambda"] for Y in (pf, nf, ns)])
    number(w, "ST of lambda, three outcomes, largest", lam.ST.max(), 0.01)
    plam = pd.concat([paw[Y][paw[Y].param == "lambda"] for Y in (pf, nf, ns)])
    both(w, "T of lambda, three outcomes", (plam["T"].min(), plam["T"].max()), 0.07, 0.10)
    both(w, "ST of m, share of financial defaults", rng(sob, pf, "m", "ST"), 0.02, 0.03)
    number(w, "T_m above the threshold at every scenario, share of financial defaults (1 = yes)", float(paw[pf][paw[pf].param == "m"].sig.all()), 1, 0)
    both(w, "T of m, share of financial defaults", rng(paw, pf, "m", "T"), 0.18, 0.21)
    number(w, "T_m above the threshold at every scenario, n_financial (1 = yes)", float(paw[nf][paw[nf].param == "m"].sig.all()), 1, 0)
    both(w, "T of m, n_financial", rng(paw, nf, "m", "T"), 0.20, 0.22)
    both(w, "T of m, n_strategic", rng(paw, ns, "m", "T"), 0.14, 0.16)
    number(w, "scenarios with T_m above the threshold, n_strategic", paw[ns][paw[ns].param == "m"].sig.sum(), 1, 0)
    both(w, "ST of m, n_financial", rng(sob, nf, "m", "ST"), 0.20, 0.22)
    both(w, "ST of varepsilon, n_financial", rng(sob, nf, "varepsilon", "ST"), 0.18, 0.20)
    number(w, "S1 of m and varepsilon, n_financial, at most 0.04 (1 = yes)",
           float(sob[nf][sob[nf].param.isin(["m", "varepsilon"])].S1.max() <= 0.04), 1, 0)
    number(w, "T_varepsilon below the threshold, both counts (1 = yes)",
           float(not pd.concat([paw[Y][paw[Y].param == "varepsilon"] for Y in (nf, ns)]).sig.any()), 1, 0)
    both(w, "T of varepsilon, n_financial", rng(paw, nf, "varepsilon", "T"), 0.12, 0.12)
    for p, t in [("delta", 0.19), ("kappa", 0.17)]:
        number(w, f"T_{p} above the threshold at every scenario, n_strategic (1 = yes)", float(paw[ns][paw[ns].param == p].sig.all()), 1, 0)
        both(w, f"T of {p}, n_strategic", rng(paw, ns, p, "T"), t, t)
        both(w, f"ST of {p}, n_strategic", rng(sob, ns, p, "ST"), 0.22, 0.22)
    both(w, "S1 of kappa, n_strategic", rng(sob, ns, "kappa", "S1"), 0.13, 0.13)
    k = sob[ns][sob[ns].param == "kappa"]
    number(w, "S1 interval of kappa clears zero, n_strategic (1 = yes)", float(((k.S1 - k.S1_conf) > 0).all()), 1, 0)
    dk = paw[nf][paw[nf].param.isin(["delta", "kappa"])]
    number(w, "T_delta and T_kappa below the threshold, n_financial (1 = yes)", float(not dk.sig.any()), 1, 0)
    both(w, "T of delta and kappa, n_financial", (dk["T"].min(), dk["T"].max()), 0.13, 0.14)
    sk = sob[nf][sob[nf].param.isin(["delta", "kappa"])]
    both(w, "ST of delta and kappa, n_financial", (sk.ST.min(), sk.ST.max()), 0.14, 0.17)

    w = "SI Distress months"
    both(w, "S1 of beta1, mean distress months", rng(sob, mm, "beta1", "S1"), 0.30, 0.33)
    both(w, "T of beta1, mean distress months", rng(paw, mm, "beta1", "T"), 0.30, 0.30)
    both(w, "ST of beta2, mean distress months", rng(sob, mm, "beta2", "ST"), 0.37, 0.37)
    both(w, "S1 of beta2, mean distress months", rng(sob, mm, "beta2", "S1"), 0.05, 0.09)
    clear = sob[mm].assign(c=sob[mm].S1 - sob[mm].S1_conf > 0).groupby("param").c.sum()
    number(w, "parameters other than beta1 whose S1 interval clears zero at every scenario (kappa only)",
           float(set(clear[clear == 5].index) - {"beta1"} == {"kappa"}), 1, 0)
    both(w, "S1 of kappa, mean distress months", rng(sob, mm, "kappa", "S1"), 0.08, 0.08)
    for p in ("beta1", "kappa"):
        number(w, f"T_{p} above the threshold at every scenario (1 = yes)", float(paw[mm][paw[mm].param == p].sig.all()), 1, 0)
    both(w, "T of kappa, mean distress months", rng(paw, mm, "kappa", "T"), 0.16, 0.17)
    number(w, "scenarios with T_beta3 above the threshold", paw[mm][paw[mm].param == "beta3"].sig.sum(), 4, 0)


# ─── Results and SI: foreclosure cascades ────────────────────────────────────────────────

def cascades():
    ps = load_paired_seed(vartheta=0.05)
    base = ps[(ps.kappa == 10) & (ps.beta2 == 0.5)]
    s0 = base[base.setting == 0]
    t0 = summarise(s0, ["delta"], ["R_net", "R_gross"]).merge(
        summarise(s0.assign(ratio=s0.rate_tot / s0.rate0), ["delta"], ["ratio"]), on="delta").set_index("delta")
    w = "Main text, Fig. 4"
    number(w, "multiplier M at S0, delta = 0.15", t0.ratio[0.15], 1.50)
    number(w, "multiplier M at S0, delta = 0.2", t0.ratio[0.2], 4.59)
    flooded, s0r, _ = r_by_scenario(ps, stat="M_net")
    m15 = flooded[np.isclose(flooded.delta, 0.15)]
    number(w, "multiplier M at delta = 0.15, lowest over S1-S4", m15.R_net.min(), 1.74)
    number(w, "multiplier M at delta = 0.15, highest over S1-S4", m15.R_net.max(), 2.02)

    methods = compare_methods()
    tree = methods[methods.method == "tree"].set_index(["group", "delta"]).share
    for g, d, printed in [("S0", 0.125, 8), ("S0", 0.2, 77), ("S0", 0.15, 31), ("S4", 0.15, 45)]:
        number(w, f"share of foreclosures in large cascades, {g}, delta = {d} (%)", 100 * tree.loc[(g, d)], printed, 0)
    pk = methods[(methods.delta >= 0.1125 - 1e-9) & methods.method.isin(["tree", "comp"])].sweep_peak
    number("SI Cascade sizes", "peak of the large-cascade mode, lowest", pk.min(), 113, 0)
    number("SI Cascade sizes", "peak of the large-cascade mode, highest", pk.max(), 175, 0)
    wide = methods[methods.delta > 0].pivot_table(index=["group", "delta"], columns="method", values="share")
    number("SI Cascade sizes", "largest difference of the attribution-free share from either reconstruction (pp, within 2)",
           float(100 * max((wide.af - wide.tree).abs().max(), (wide.af - wide.comp).abs().max()) <= 2), 1, 0)
    af_sizes, af_runs = load_attribution_free()
    _, _, af_peaks = size_density(af_sizes, af_runs[["run_id", "setting", "lambda", "delta", "seed"]])
    number("Main text, Fig. 4", "attribution-free large-cascade mode at delta = 0 (scenario groups with one)",
           af_peaks[(af_peaks.delta == 0)].sweep_peak.notna().sum(), 0, 0)
    sizes, meta = load_cascade_sizes()
    _, _, peaks = size_density(sizes, meta)
    sp = sparks_and_sweeps(sizes, meta, peaks)
    sp = sp[sp.method == "tree"].set_index(["group", "delta"])
    number("SI Cascade sizes", "large cascades per run, S4 / S0, delta = 0.15 (about seven)",
           sp.sweeps_per_run[("S4", 0.15)] / sp.sweeps_per_run[("S0", 0.15)], 7, 0)
    number("SI Cascade sizes", "large cascades in the 20 S0 runs, delta = 0.125", 20 * sp.sweeps_per_run[("S0", 0.125)], 4, 0)

    w = "SI Cascade method"
    low = t0[(t0.index > 0) & (t0.index <= 0.125)]
    number(w, "S0: R_net interval contains zero for every delta <= 0.125 (1 = yes)",
           float(((low.R_net_lo <= 0) & (low.R_net_hi >= 0)).all()), 1, 0)
    number(w, "S0: R_gross at delta = 0.125", t0.R_gross[0.125], 0.15)
    rn, _, w_seed = r_by_scenario(ps, stat="R_net")
    pv = (w_seed.groupby(["setting", "delta"]).dR_net.apply(lambda x: stats.ttest_1samp(x, 0).pvalue)
          .rename("p").reset_index())
    rn = rn.merge(pv, on=["setting", "delta"])
    rn = rn[rn.delta > 0].copy()
    order = np.argsort(rn.p.to_numpy())
    holm = np.zeros(len(rn), bool)
    for rank, i in enumerate(order):
        if rn.p.to_numpy()[i] <= 0.05 / (len(rn) - rank):
            holm[i] = True
        else:
            break
    rn["holm"] = holm
    number(w, "combinations of scenario and delta > 0", len(rn), 44, 0)
    early = rn[rn.delta <= 0.05 + 1e-9]
    number(w, "Delta R_net interval contains zero for every delta <= 0.05 (1 = yes)",
           float(((early.dR_net_lo <= 0) & (early.dR_net_hi >= 0)).all()), 1, 0)
    mid = rn[(rn.delta >= 0.1 - 1e-9) & (rn.delta <= 0.175 + 1e-9)]
    number(w, "combinations with delta in [0.1, 0.175]", len(mid), 20, 0)
    number(w, "of which the Delta R_net interval lies above zero", (mid.dR_net_lo > 0).sum(), 17, 0)
    number(w, "of which Holm-significant (correction over all 44 combinations)", mid.holm.sum(), 5, 0)
    top = rn.loc[rn.dR_net.idxmax()]
    number(w, "largest Delta R_net: scenario", top.setting, 4, 0)
    number(w, "largest Delta R_net: delta", top.delta, 0.15)
    number(w, "R_net at S4, delta = 0.15", top.R_net, 0.48)
    number(w, "R_net at S0, delta = 0.15", top.R_net_S0, 0.23)
    number(w, "Delta R_net negative at delta = 0.25 in every scenario (1 = yes)", float((rn[np.isclose(rn.delta, 0.25)].dR_net < 0).all()), 1, 0)

    ps_all = load_paired_seed()
    def r_curve(vt, kappa, beta2):
        d = ps_all[(ps_all.vartheta == vt) & (ps_all.kappa == kappa) & (ps_all.beta2 == beta2) & (ps_all.delta > 0)]
        wv = pd.concat([d[d.setting == 0].groupby(["delta", "seed"]).R_net.mean().rename("S0"),
                        d[d.setting > 0].groupby(["delta", "seed"]).R_net.mean().rename("flooded")], axis=1).reset_index()
        wv["dR"] = wv.flooded - wv.S0
        return summarise(wv, ["delta"], ["dR"])
    c3 = r_curve(0.05, 3, 0.5)
    number(w, "kappa = 3: Delta R_net interval contains zero at every delta (1 = yes)",
           float(((c3.dR_lo <= 0) & (c3.dR_hi >= 0)).all()), 1, 0)
    c7 = r_curve(0.05, 10, 0.3)
    number(w, "gap 0.7: Delta R_net interval above zero only at delta = 0.175 (1 = yes)",
           float(c7[c7.dR_lo > 0].delta.round(4).tolist() == [0.175]), 1, 0)
    cors = []
    for _, g in ps[(ps.setting > 0) & (ps.delta > 0)].groupby(["kappa", "beta2", "delta"]):
        c = np.corrcoef(g.pivot_table(index="seed", columns=["setting", "lambda"], values="R_net").to_numpy().T)
        cors.append(c[np.triu_indices_from(c, 1)].mean())
    number("SI Fig. 10 caption", "mean pairwise correlation of flooded runs sharing a seed, lowest", min(cors), -0.06)
    number("SI Fig. 10 caption", "mean pairwise correlation of flooded runs sharing a seed, highest", max(cors), 0.08)

    pct = flooded.set_index(["delta", "setting"]).dR_net
    for d, lo, hi in [(0.125, 12, 27), (0.15, 21, 43), (0.25, -27, -18)]:
        x = pct.xs(d, level="delta")
        number(w, f"% change of M under floods at delta = {d}, lowest", x.min(), lo, 0)
        number(w, f"% change of M under floods at delta = {d}, highest", x.max(), hi, 0)

    # seed as unit (Statistics section)
    w = "Statistics, seed intervals"
    m = ps[np.isclose(ps.delta, 0.15) & (ps.kappa == 10) & (ps.beta2 == 0.5)]
    for s, printed in [(1, 1.74), (4, 2.02)]:
        number(w, f"multiplier M S{s}, delta = 0.15, mean over seeds", t_mean(m[m.setting == s].groupby("seed").M_net.mean())[0], printed)


def main():
    flood_scenarios_and_maps()
    baseline = initialization()
    admission()
    zeta_scale()
    calibration(baseline)
    rates()
    maps_and_composition()
    timing()
    sensitivity()
    cascades()

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
