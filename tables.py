"""Results tables of the paper and the SI, as LaTeX tabulars.

    Table 1       monthly probability of each flood-damage tier under S0 to S4
    SI Table 6    calibration: KS distance of every parameter between retained and all
                  combinations, bootstrap interval, max-T threshold
    SI Table 8    rank of every parameter by its PAWN index (mean over the scenarios) at
                  J = 5, 10 and 20 conditioning intervals
    SI Table 9    cascades per run, large cascades per 1,000 cascades and per run, S0, S1, S4
    SI Table 10   S0: share of foreclosures in large cascades, R_gross, R_net, M, mean tree size

Run:    uv run python tables.py
Writes: output/tables/table_1.tex, table_S6.tex, table_S8.tex, table_S9.tex, table_S10.tex
"""
import os

import numpy as np
import pandas as pd
from SALib.analyze import pawn
from scipy.stats import genextreme

from utils import (ABC_PARAMS, GEV_SCENARIOS, TABLES, abc_bootstrap, abc_results, compare_methods, gsa_outcome,
                   gsa_problem, load_cascade_sizes, load_paired_seed, pawn_design, size_density,
                   sparks_and_sweeps, summarise)

TIERS = [("No flood", "none"), ("Minor", r"$R_{grote}$"), ("Moderate", r"$R_{middel}$"),
         ("Severe", r"$R_{klein}$"), ("Catastrophic", r"$R_{eklein}$")]
SYMBOL = {"kappa": r"\kappa", "beta1": r"\beta_1", "beta2": r"\beta_2", "beta3": r"\beta_3", "lambda": r"\lambda",
          "delta": r"\delta", "eta": r"\eta", "vartheta": r"\vartheta", "m": "m", "varepsilon": r"\varepsilon",
          "exp_lb": r"\mathit{exp}_{lb}"}


def write(name, lines):
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, f"{name}.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(path)}")
    print("\n".join(lines) + "\n")


def table_1():
    """Probability (%) that a month falls in each damage tier: sev <= 0, (0, 30], (30, 300],
    (300, 3000], > 3000."""
    cuts = [0, 30, 300, 3000]
    probs = {0: [100.0, 0, 0, 0, 0]}
    for s, (mu, sigma, xi) in GEV_SCENARIOS.items():
        c = genextreme.cdf(cuts, c=-xi, loc=mu, scale=sigma)
        probs[s] = list(100 * np.diff(np.r_[0, c, 1]))
    fmt = lambda v: "{$<$0.1}" if 0 < v < 0.05 else f"{v:.1f}"
    rows = [rf"\textbf{{{name}}} & {m} & " + " & ".join(fmt(probs[s][i]) for s in range(5)) + r" \\"
            for i, (name, m) in enumerate(TIERS)]
    write("table_1", [r"\toprule",
                      r"{Damage tier} & {Map} & {$\mathcal{S}_0$} & {$\mathcal{S}_1$} & {$\mathcal{S}_2$} & {$\mathcal{S}_3$} & {$\mathcal{S}_4$} \\",
                      r"\midrule", *rows, r"\bottomrule"])


def table_s6():
    s, pos, accepted, band, ks = abc_results()
    main, strict = ks[ks.tol == 0.05].set_index("param"), ks[ks.tol == 0.01].set_index("param")
    boot = abc_bootstrap(pos, accepted[0.05], main.ks_p95.iloc[0]).set_index("param")
    num = lambda v, bold: rf"$\mathbf{{{v:.2f}}}$" if bold else f"${v:.2f}$"
    rows = [rf"${SYMBOL[p]}$ & {num(main.loc[p, 'ks'], main.loc[p, 'constrained'])} & "
            rf"$[{boot.loc[p, 'ks_lo']:.2f},\,{boot.loc[p, 'ks_hi']:.2f}]$ & ${boot.loc[p, 'p_above']:.2f}$ & "
            rf"{num(strict.loc[p, 'ks'], strict.loc[p, 'constrained'])} \\" for p in ABC_PARAMS]
    write("table_S6", [r"\begin{tabular}{l c c c c}", r"\toprule",
                       r"Parameter & KS ($5\%$) & Bootstrap $90\%$ interval & Share above threshold & KS ($1\%$) \\",
                       r"\midrule", *rows, r"\midrule",
                       rf"Max-$T$ threshold & ${main.ks_p95.iloc[0]:.2f}$ & & & ${strict.ks_p95.iloc[0]:.2f}$ \\",
                       r"\bottomrule", r"\end{tabular}"])


def table_s8():
    """Rank (1 = largest) of the PAWN index averaged over the five scenarios; bold where the
    rank is 1 or 2 at every J."""
    problem, X = gsa_problem(), pawn_design()
    names = problem["names"]
    ranks = {}
    for Y in ("perc_fin", "n_financial", "n_strategic"):
        for J in (5, 10, 20):
            mean = np.mean([np.asarray(pawn.analyze(problem, X, gsa_outcome(s, Y)[:len(X)], S=J, seed=0)["median"], float)
                            for s in range(5)], axis=0)
            ranks[(Y, J)] = pd.Series(mean, index=names).rank(ascending=False).astype(int)
    rows = []
    for p in names:
        cells = []
        for Y in ("perc_fin", "n_financial", "n_strategic"):
            bold = all(ranks[(Y, J)][p] <= 2 for J in (5, 10, 20))
            cells += [(r"\bfseries " if bold else "") + str(ranks[(Y, J)][p]) for J in (5, 10, 20)]
        rows.append(rf"${SYMBOL[p]}$ & " + " & ".join(cells) + r" \\")
    write("table_S8", [r"\toprule",
                       r" & \multicolumn{3}{c}{\% Financial defaults} & \multicolumn{3}{c}{$n_{\text{financial}}$} & \multicolumn{3}{c}{$n_{\text{strategic}}$} \\",
                       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
                       r"Parameter & {$J{=}5$} & {$J{=}10$} & {$J{=}20$} & {$J{=}5$} & {$J{=}10$} & {$J{=}20$} & {$J{=}5$} & {$J{=}10$} & {$J{=}20$} \\",
                       r"\midrule", *rows, r"\bottomrule"])


def table_s9():
    sizes, meta = load_cascade_sizes()
    _, _, peaks = size_density(sizes, meta)
    sp = sparks_and_sweeps(sizes, meta, peaks)
    sp = sp[sp.method == "tree"].set_index(["delta", "group"])
    rows = []
    for d in (0.125, 0.15, 0.2, 0.25):
        r = sp.xs(d, level="delta")
        rows.append(rf"${d:g}$ & " + " & ".join(f"${r.loc[g, 'sparks_per_run']:.0f}$" for g in ("S0", "S1", "S4")) + " & "
                    + " & ".join(f"${r.loc[g, 'sweeps_per_1000_sparks']:.1f}$" for g in ("S0", "S1", "S4")) + " & "
                    + " & ".join(f"${r.loc[g, 'sweeps_per_run']:.2f}$" for g in ("S0", "S1", "S4")) + r" \\")
    write("table_S9", [r"\begin{tabular}{l rrr rrr rrr}", r"\toprule",
                       r" & \multicolumn{3}{c}{Cascades per run} & \multicolumn{3}{c}{Large cascades per $1{,}000$ cascades} & \multicolumn{3}{c}{Large cascades per run} \\",
                       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
                       r"$\delta$ & $\mathcal{S}_0$ & $\mathcal{S}_1$ & $\mathcal{S}_4$ & $\mathcal{S}_0$ & $\mathcal{S}_1$ & $\mathcal{S}_4$ & $\mathcal{S}_0$ & $\mathcal{S}_1$ & $\mathcal{S}_4$ \\",
                       r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


def table_s10():
    ps = load_paired_seed(vartheta=0.05)
    s0 = ps[(ps.kappa == 10) & (ps.beta2 == 0.5) & (ps.setting == 0)]
    t0 = summarise(s0, ["delta"], ["R_net", "R_gross"]).merge(
        summarise(s0.assign(ratio=s0.rate_tot / s0.rate0), ["delta"], ["ratio"]), on="delta").set_index("delta")
    sizes, meta = load_cascade_sizes()
    z = sizes[sizes.method == "tree"].merge(meta[meta.setting == 0][["run_id", "delta"]], on="run_id")
    mean_tree = z.groupby("delta").apply(lambda x: (x["size"] * x["count"]).sum() / x["count"].sum(), include_groups=False)
    share = compare_methods()
    share = share[(share.method == "tree") & (share.group == "S0")].set_index("delta").share
    rows = []
    for d in (0.125, 0.15, 0.175, 0.2, 0.25):
        r = t0.loc[d]
        rows.append(rf"${d:g}$ & ${100 * share.loc[d]:.0f}\%$ & ${r.R_gross:.2f}$ & ${r.R_net:.2f}$ & "
                    rf"${r.R_net / r.R_gross:.2f}$ & ${r.ratio:.2f}$ & ${mean_tree.loc[d]:.2f}$ \\")
    write("table_S10", [r"\begin{tabular}{l rrrrrr}", r"\toprule",
                        r"$\delta$ & Share in large cascades & $R_{\text{gross}}$ & $R_{\text{net}}$ & $R_{\text{net}}/R_{\text{gross}}$ & $M$ & Mean tree size \\",
                        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    table_1()
    table_s6()
    table_s8()
    table_s9()
    table_s10()
