"""SI Fig. 10: net reproduction number of the foreclosure feedback with and without floods, by
radius kappa, gap beta1 - beta2, damage share lambda and flood-induced unemployment vartheta.

Top: R_net against delta at S0 (open squares) and under S1-S4 (filled circles), means with 95%
Student-t intervals, each run one observation. In the lambda and vartheta columns S0 does not
depend on the level and is drawn once, in black. Bottom: Delta R_net = R_net(S1-S4) - R_net(S0),
per seed (flooded runs of the seed averaged, minus the S0 run of the same seed), 95% bands over
20 seeds. Dotted line: R_net = 0.1.

Run:    python figure_SI10.py
Writes: output/figures/figure_S10.png
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from utils import FS, curve, level_axis, load_paired_seed, points, save_figure, style_axes, summarise

VARTHETA_BASE = 0.05
R_REF = 0.1
KAPPA_COLORS = {3: "#9e9e9e", 7: "#74a9cf", 10: "#2978b5", 15: "#ef6548", 18: "#b2182b"}
GAP_COLORS = {0.5: "#2978b5", 0.6: "#e6a117", 0.7: "#41ab5d"}
LAMBDA_COLORS = {0.25: "#fdae61", 0.5: "#d7191c", 0.75: "#7b3294"}
VARTHETA_COLORS = {0.03: "#bcbddc", 0.05: "#807dba", 0.13: "#3f007d"}
FLOODED = r"$\mathcal{S}_1$-$\mathcal{S}_4$"

# columns: (title, [(label, kappa, beta2, lambda, vartheta, colour), ...]); lambda None = averaged
COLUMNS = [
    (r"Radius $\kappa$ (gap 0.5)",
     [(rf"$\kappa = {k}$", k, 0.5, None, VARTHETA_BASE, c) for k, c in KAPPA_COLORS.items()]),
    (r"Gap $\beta_1 - \beta_2$ ($\kappa = 10$)",
     [(rf"gap ${g:g}$", 10, round(1 - g, 1), None, VARTHETA_BASE, c) for g, c in GAP_COLORS.items()]),
    (r"Damage share $\lambda$ ($\kappa = 10$, gap 0.5)",
     [(rf"$\lambda = {lam:g}$", 10, 0.5, lam, VARTHETA_BASE, c) for lam, c in LAMBDA_COLORS.items()]),
    (r"Flood unemployment $\vartheta$ ($\kappa = 10$, gap 0.5)",
     [(rf"$\vartheta = {v:g}$", 10, 0.5, None, v, c) for v, c in VARTHETA_COLORS.items()]),
]


def r_curve(ps, kappa, beta2, lam=None):
    """R_net at S0 and under floods (S1-S4, and lambda unless fixed), each run one observation;
    Delta R_net per seed. ps is sliced to one vartheta."""
    d = ps[(ps.kappa == kappa) & (ps.beta2 == beta2) & (ps.delta > 0)]
    fl = d[d.setting > 0] if lam is None else d[(d.setting > 0) & (d["lambda"] == lam)]
    s0 = summarise(d[d.setting == 0].rename(columns={"R_net": "S0"}), ["delta"], ["S0"])
    flr = summarise(fl.rename(columns={"R_net": "flooded"}), ["delta"], ["flooded"])
    w = pd.concat([d[d.setting == 0].groupby(["delta", "seed"])["R_net"].mean().rename("S0"),
                   fl.groupby(["delta", "seed"])["R_net"].mean().rename("flooded")], axis=1).reset_index()
    w["dR"] = w.flooded - w.S0
    return s0.merge(flr, on="delta").merge(summarise(w, ["delta"], ["dR"]), on="delta")


def main():
    ps = load_paired_seed()
    F = FS + 10                          # four wide panels, shrunk to the text width
    fig, axes = plt.subplots(2, len(COLUMNS), figsize=(11 * len(COLUMNS), 18), sharex=True, sharey="row")
    for j, (title, items) in enumerate(COLUMNS):
        ax_r, ax_d = axes[0, j], axes[1, j]
        dodge = 0.0012                   # x offset between the series of one panel
        s0_once = items[0][3] is not None or items[0][4] != items[-1][4]     # lambda or vartheta column
        for i, (lab, k, b2, lam, vt, c) in enumerate(items):
            s = r_curve(ps[ps.vartheta == vt], k, b2, lam)
            d0, d1 = 2 * i - len(items) + 0.5, 2 * i - len(items) + 1.5
            if not s0_once:
                points(ax_r, s, "delta", "S0", c, None, dx=d0 * dodge, marker="s", markerfacecolor="white", markeredgewidth=2.5)
            elif i == 0:
                points(ax_r, s, "delta", "S0", "black", None, dx=d0 * dodge, marker="s", markerfacecolor="white", markeredgewidth=2.5)
            points(ax_r, s, "delta", "flooded", c, None, dx=d1 * dodge)
            curve(ax_d, s, "dR", c)
        ax_r.axhline(R_REF, color="grey", linestyle=":", linewidth=2)
        ax_r.set_title(title, fontsize=F, fontweight="bold")
        h = [Line2D([], [], color=c, linewidth=4, label=lab) for lab, *_, c in items]
        if s0_once:
            h.append(Line2D([], [], color="black", linestyle="", marker="s", markerfacecolor="white", markeredgewidth=2.5,
                            markersize=11, label=r"$\mathcal{S}_0$"))
        ax_r.legend(handles=h, fontsize=F - 3, framealpha=0, loc="upper left")
        style_axes(ax_r, "", r"$R_{\mathrm{net}}$" if j == 0 else "")
        level_axis(ax_r, multiplier=False)
        style_axes(ax_d, r"$\delta$", r"$\Delta R_{\mathrm{net}}$ (floods $-$ $\mathcal{S}_0$)" if j == 0 else "", zero=True)
        for ax in (ax_r, ax_d):
            ax.xaxis.label.set_fontsize(F)
            ax.yaxis.label.set_fontsize(F)
            ax.tick_params(labelsize=F - 3)
    axes[0, 0].text(0.003, R_REF + 0.015, "threshold", fontsize=F - 10, color="grey")
    h = [Line2D([], [], color="black", linestyle="", marker="s", markerfacecolor="white", markeredgewidth=2.5, markersize=11,
                label=r"$\mathcal{S}_0$ ($R_{\mathrm{net}}$)"),
         Line2D([], [], color="black", linestyle="", marker="o", markersize=11, label=FLOODED + r" ($R_{\mathrm{net}}$)")]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=F + 3, framealpha=0)
    save_figure(fig, "figure_S10", dpi=200)


if __name__ == "__main__":
    main()
