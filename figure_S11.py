"""SI Fig. 11: distress months among foreclosed households, including forborne months and the
terminal foreclosure, by channel (rows) and by flood scenario, gap beta1 - beta2, debt pressure
beta3 and damage share lambda (columns). Legends: median distress months and the share of
foreclosures in the channel. Flood scenarios S1-S4 pooled except in the first column.

Run:    uv run python figure_S11.py
Writes: output/figures/figure_S11.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import FS, SETTING_COLORS, TIMING_CHANNELS, load_timing, save_figure, style_density_axes

BINS = np.arange(0, 76, 2)


def main():
    fcl_gap, fcl_b3 = load_timing("gap"), load_timing("beta3")
    fcl = fcl_gap[fcl_gap.gap == 0.5].reset_index(drop=True)
    flooded = fcl[fcl.setting > 0]
    b3 = pd.concat([flooded, fcl_b3[fcl_b3.setting > 0]])
    gaps = sorted(fcl_gap.gap.unique())
    gap_cols = dict(zip(gaps, plt.cm.viridis(np.linspace(0.85, 0.1, len(gaps)))))
    columns = [
        ("Flood scenario", fcl, "setting", sorted(fcl.setting.unique()), SETTING_COLORS,
         lambda v: rf"$\mathcal{{S}}_{v}$"),
        (r"Gap $\beta_1-\beta_2$", fcl_gap[fcl_gap.setting > 0], "gap", gaps, gap_cols, lambda v: f"{v:g}"),
        (r"Debt pressure $\beta_3$", b3, "beta3", [0.5, 1.0, 2.0], {0.5: "#c6dbef", 1.0: "#4292c6", 2.0: "#08306b"},
         lambda v: f"{v:g}"),
        (r"Damage share $\lambda$", flooded[flooded["lambda"].isin([0.25, 0.5])], "lambda", [0.25, 0.5],
         {0.25: "#fdae61", 0.5: "#d7191c"}, lambda v: f"{v:g}"),
    ]
    fig, axes = plt.subplots(2, len(columns), figsize=(9 * len(columns), 15), sharex=True, sharey="row")
    for j, (title, d, key, levels, cols, lab) in enumerate(columns):
        for i, (reason, (cname, _)) in enumerate(TIMING_CHANNELS.items()):
            ax = axes[i, j]
            for v in levels:
                dv = d[d[key] == v]
                n = dv.loc[dv.default_reason == reason, "n_dis"]
                if n.empty:
                    continue
                share = 100 * len(n) / len(dv)
                ax.hist(n, bins=BINS, density=True, histtype="step", linewidth=3.5, color=cols[v],
                        label=f"{lab(v)}: {n.median():.0f} mo, {share:.0f}%")
                ax.axvline(n.median(), color=cols[v], linestyle=":", linewidth=2.5)
            ax.legend(fontsize=FS - 9, framealpha=0, loc="upper right")
            style_density_axes(ax, "Distress months" if i == 1 else "", 75)
            if j > 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(f"{cname.capitalize()}\nDensity", fontsize=FS, fontweight="bold")
            if i == 0:
                ax.set_title(title, fontsize=FS, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figure_S11", dpi=200)


if __name__ == "__main__":
    main()
