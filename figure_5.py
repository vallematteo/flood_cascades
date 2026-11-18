"""Fig. 5: mortgage age at foreclosure.

(a) distribution of the age at foreclosure by flood scenario, both channels; (b) by channel,
flood scenarios S1-S4 pooled. Dotted lines: medians. Calibrated parametrisation, beta1 = 1,
beta2 = 0.5, beta3 = 1, delta = 0, lambda in {0.25, 0.5, 0.75}, 10 seeds; foreclosures after
the warm-up.

Run:    python figure_5.py
Writes: output/figures/figure_5.png
"""
import matplotlib.pyplot as plt
import numpy as np

from utils import FS, SETTING_COLORS, TIMING_CHANNELS, load_timing, save_figure, style_density_axes

BINS = np.arange(0, 126, 5)


def main():
    fcl = load_timing("gap")
    fcl = fcl[fcl.gap == 0.5].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(21, 8))

    ax = axes[0]
    for s in sorted(fcl.setting.unique()):
        t = fcl.loc[fcl.setting == s, "t"]
        ax.hist(t, bins=BINS, density=True, histtype="step", linewidth=3.5, color=SETTING_COLORS[s],
                label=rf"$\mathcal{{S}}_{s}$")
        ax.axvline(t.median(), color=SETTING_COLORS[s], linestyle=":", linewidth=2.5)
    ax.legend(fontsize=FS - 8, framealpha=0, ncol=2)

    ax = axes[1]
    flooded = fcl[fcl.setting > 0]
    for reason, (name, color) in TIMING_CHANNELS.items():
        t = flooded.loc[flooded.default_reason == reason, "t"]
        ax.hist(t, bins=BINS, density=True, histtype="stepfilled", alpha=0.35, color=color)
        ax.hist(t, bins=BINS, density=True, histtype="step", linewidth=3.5, color=color,
                label=f"{name} (median {t.median():.0f})")
        ax.axvline(t.median(), color=color, linestyle=":", linewidth=2.5)
    ax.legend(fontsize=FS - 8, framealpha=0)

    for ax in axes:
        style_density_axes(ax, "Household age at foreclosure", 120)
    fig.tight_layout()
    save_figure(fig, "figure_5", dpi=200)


if __name__ == "__main__":
    main()
