"""SI Fig. 1: densities of the flood-severity distributions of the scenarios S1 to S4.

GEV densities with the parameters of preparation/gev/gev_scenario_parameters.py, with an
inset on the upper tail. No simulation data.

Run:    python reproduce/figure_SI1.py
Writes: output/figures/figure_S1.png
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genextreme

from utils import FS, GEV_SCENARIOS, SETTING_COLORS, save_figure

TAIL = (320, 390)        # x range of the inset


def main():
    x = np.linspace(-10, 400, 4000)
    pdfs = {s: genextreme.pdf(x, c=-xi, loc=mu, scale=sigma)       # scipy's shape is c = -xi
            for s, (mu, sigma, xi) in GEV_SCENARIOS.items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    for s, pdf in pdfs.items():
        ax.plot(x, pdf, color=SETTING_COLORS[s], lw=2.5, label=fr"$\mathcal{{S}}_{s}$")
    ax.set_xlim(-10, 400)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Flood severity", fontsize=FS, fontweight="bold")
    ax.set_ylabel("Density $f(x)$", fontsize=FS, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FS - 6)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(FS - 8)
    ax.legend(fontsize=FS - 8, framealpha=0, title="Flood scenario", title_fontsize=FS - 8, loc="upper right")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # inset on the upper tail, y range fitted to the tail values
    mask = (x >= TAIL[0]) & (x <= TAIL[1])
    ymax = 1.15 * max(pdf[mask].max() for pdf in pdfs.values())
    ax_in = ax.inset_axes([0.30, 0.42, 0.33, 0.45])
    for s, pdf in pdfs.items():
        ax_in.plot(x[mask], pdf[mask], color=SETTING_COLORS[s], lw=2.5)
    ax_in.set_xlim(*TAIL)
    ax_in.set_ylim(0, ymax)
    ax_in.set_xticks(TAIL)
    ax_in.tick_params(axis="both", labelsize=FS - 10)
    ax_in.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_in.yaxis.get_offset_text().set_fontsize(FS - 12)
    ax_in.grid(axis="y", ls="--", alpha=0.4)
    ax.indicate_inset_zoom(ax_in, edgecolor="black", alpha=0.5)

    fig.tight_layout()
    save_figure(fig, "figure_S1", dpi=300)


if __name__ == "__main__":
    main()
