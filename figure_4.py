"""Fig. 4: foreclosure feedback and default cascades against the spillover intensity delta.

(a) Foreclosure-rate multiplier M = 1/(1 - R_net) by flood scenario (log axis): means over 20
    seeds with 95% Student-t intervals, lambda averaged within seed.
(b)-(d) Share of foreclosures by cascade size at S0, S1 and S4 (weighted-tree reconstruction):
    Gaussian kernel of bandwidth 0.3 in log size, each violin scaled to its own maximum;
    horizontal tick = valley between isolated foreclosures and large cascades; number above =
    share of foreclosures in large cascades.
Slice: calibrated parametrisation, kappa = 10, beta1 - beta2 = 0.5, vartheta = 0.05.

Run:    python figure_4.py
Writes: output/figures/figure_4.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (FS, SETTING_COLORS, delta_axis, delta_positions, level_axis, load_cascade_sizes,
                   load_paired_seed, points, r_by_scenario, save_figure, scenario_label, size_density,
                   style_axes)

VIOLIN_COLORS = {"S0": "#636363", "S1": "#9ecae1", "S4": "#08519c"}
VIOLIN_YTICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
GROUP_LABELS = {"S0": r"$\mathcal{S}_0$", "S1": r"$\mathcal{S}_1$", "S4": r"$\mathcal{S}_4$"}


def multiplier_panel(ax, flooded, s0r, dodge=0.15):
    deltas, fl, s0 = delta_positions(flooded, s0r)
    points(ax, s0, "x", "M_net", SETTING_COLORS[0], scenario_label(0), dx=-2 * dodge)
    for s in range(1, 5):
        points(ax, fl[fl.setting == s], "x", "M_net", SETTING_COLORS[s], scenario_label(s), dx=(s - 2) * dodge)
    ax.set_title(r"(a) $M$", fontsize=FS, fontweight="bold")
    style_axes(ax, r"$\delta$", r"$M$")
    level_axis(ax, multiplier=True)
    ax.legend(fontsize=FS, framealpha=0)
    delta_axis(ax, deltas)


def violin_panel(ax, grid, dens, peaks, group, deltas, title, ylabel, half_width=0.42):
    c = VIOLIN_COLORS[group]
    y = np.exp(grid)
    pk = peaks[(peaks.method == "tree") & (peaks.group == group)].set_index("delta").reindex(deltas)
    for i, d in enumerate(deltas):
        k = dens[("tree", group, d)]
        w = half_width * k / k.max()
        keep = w > 1e-3 * half_width
        ax.fill_betweenx(y[keep], i - w[keep], i + w[keep], color=c, alpha=0.55, linewidth=0)
        ax.plot(np.r_[i - w[keep]], y[keep], color=c, lw=1.5)
        ax.plot(np.r_[i + w[keep]], y[keep], color=c, lw=1.5)
        r = pk.loc[d]
        if pd.notna(r.sweep_peak):
            ax.plot([i - half_width * 0.6, i + half_width * 0.6], [r.valley] * 2, color="black", lw=2, zorder=5)
        if r.sweep_share > 0:
            ax.annotate(f"{100 * r.sweep_share:.0f}%" if r.sweep_share >= 0.01 else "<1%", (i, 1300),
                        ha="center", va="top", fontsize=FS - 10, color="#333333", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1, 1500)
    ax.set_xlim(-0.6, len(deltas) - 0.4)
    ax.set_title(title, fontsize=FS, fontweight="bold")
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([f"{d:g}" for d in deltas], rotation=60, ha="right")
    ax.tick_params(labelsize=FS - 6)
    ax.set_xlabel(r"$\delta$", fontsize=FS, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks(VIOLIN_YTICKS)
    ax.set_yticklabels([str(v) for v in VIOLIN_YTICKS])
    if ylabel:
        ax.set_ylabel("cascade size (foreclosures)", fontsize=FS, fontweight="bold")


def main():
    flooded, s0r, _ = r_by_scenario(load_paired_seed(vartheta=0.05), stat="M_net")
    grid, dens, peaks = size_density(*load_cascade_sizes())
    deltas = np.array(sorted({d for (m, _, d) in dens if m == "tree" and d >= 0.05 - 1e-9}))

    fig, axes = plt.subplots(2, 2, figsize=(26, 19))
    multiplier_panel(axes[0, 0], flooded, s0r)
    for ax in axes.flat[2:]:
        ax.sharey(axes[0, 1])
    for ax, group, letter in zip(axes.flat[1:], ["S0", "S1", "S4"], ["(b)", "(c)", "(d)"]):
        violin_panel(ax, grid, dens, peaks, group, deltas, f"{letter} {GROUP_LABELS[group]}", ylabel=(group != "S4"))
    fig.tight_layout(h_pad=3)
    save_figure(fig, "figure_4", dpi=200)


if __name__ == "__main__":
    main()
