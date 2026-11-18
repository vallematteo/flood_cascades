"""SI Fig. 12: mortgage age at foreclosure by damage share lambda in {0.25, 0.5, 0.75}, flood
scenarios S1-S4 pooled: (a) both channels, (b) strategic foreclosures. Legends: median age and
number of foreclosures.

Run:    python figure_SI12.py
Writes: output/figures/figure_S12.png
"""
import matplotlib.pyplot as plt
import numpy as np

from utils import FS, load_timing, save_figure, style_density_axes

LAMBDA_COLORS = {0.25: "#fdae61", 0.5: "#d7191c", 0.75: "#7b3294"}


def main():
    fcl = load_timing("gap")
    d = fcl[(fcl.gap == 0.5) & (fcl.setting > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(21, 8))
    for ax, (reason, title) in zip(axes, [(None, "(a) both channels"), (-1, "(b) strategic channel")]):
        sub = d if reason is None else d[d.default_reason == reason]
        for lam in sorted(sub["lambda"].unique()):
            t = sub.loc[sub["lambda"] == lam, "t"]
            ax.hist(t, bins=np.arange(0, 126, 5), density=True, histtype="step", linewidth=3.5,
                    color=LAMBDA_COLORS[lam], label=rf"$\lambda = {lam:g}$ ({t.median():.0f}, n = {len(t):,})")
            ax.axvline(t.median(), color=LAMBDA_COLORS[lam], linestyle=":", linewidth=2.5)
        ax.set_title(title, fontsize=FS, fontweight="bold")
        ax.legend(fontsize=FS - 8, framealpha=0)
        style_density_axes(ax, "Household age at foreclosure", 120)
    fig.tight_layout()
    save_figure(fig, "figure_S12", dpi=200)


if __name__ == "__main__":
    main()
