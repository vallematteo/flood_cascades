"""SI Fig. 9: change of the foreclosure-rate multiplier under floods.

Percentage change of M = 1/(1 - R_net) between each flood scenario and S0, per seed
(100 (exp(mean ln ratio) - 1), lambda averaged within seed), against delta, with 95%
Student-t bands over 20 seeds. Slice: calibrated parametrisation, kappa = 10,
beta1 - beta2 = 0.5, vartheta = 0.05.

Run:    uv run python figure_SI9.py
Writes: output/figures/figure_S9.png
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import (FS, SETTING_COLORS, curve, delta_axis, delta_positions, load_paired_seed,
                   r_by_scenario, save_figure, scenario_label, style_axes)


def main():
    flooded, s0r, _ = r_by_scenario(load_paired_seed(vartheta=0.05), stat="M_net")
    deltas, fl, _ = delta_positions(flooded, s0r)

    fig, ax = plt.subplots(figsize=(13, 8.5))
    for s in range(1, 5):
        curve(ax, fl[fl.setting == s].drop(columns="delta").rename(columns={"x": "delta"}), "dR_net", SETTING_COLORS[s])
    ax.set_title(r"$M(\mathcal{S}) / M(\mathcal{S}_0) - 1$ (%)", fontsize=FS, fontweight="bold")
    style_axes(ax, r"$\delta$", r"% change in $M$", zero=True)
    delta_axis(ax, deltas)
    ax.legend(handles=[Line2D([], [], color=SETTING_COLORS[s], marker="o", markersize=9, linewidth=3,
                              label=scenario_label(s)) for s in range(1, 5)],
              fontsize=FS, framealpha=0, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "figure_S9", dpi=200)


if __name__ == "__main__":
    main()
