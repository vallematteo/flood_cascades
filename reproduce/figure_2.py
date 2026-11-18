"""Fig. 2: calibration at the no-flood scenario.

Left: foreclosure rate and share of households in financial distress of the 1,001 parameter
combinations (mean over 10 seeds), the NHG and BKR bands, the combinations retained at the 5%
and 1% tolerances and the one inside both bands. Right: per parameter, the 5-95% spread of the
retained combinations in the sensitivity-analysis range; parameters constrained at the 5%
tolerance are starred, with the value of the combination inside both bands.

Run:    python reproduce/figure_2.py
Writes: output/figures/figure_2.png
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from utils import (ABC_PARAMS, ABC_TOLERANCES, EMPIRICAL_BANDS, FS, GREEK, abc_results, load_problem,
                   save_figure)

GREY = "#B8B8B8"
TOL_COLORS = {0.05: "#9B72B9", 0.01: "#5B2C83"}
BAND = "#2E1145"
FCL = "#E08A8F"
DIS = "#7FA7D9"
TOL_MAIN = 0.05


def fmt_bound(v):
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"


def style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=FS - 2)


def main():
    s, pos, accepted, band, ks = abc_results()

    fig = plt.figure(figsize=(28, 16))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.25)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # left: the two metrics of every combination
    (flo, fhi), (dlo, dhi) = EMPIRICAL_BANDS["rate_tot"], EMPIRICAL_BANDS["arrears_fin"]
    ax1.axvspan(flo, fhi, color=FCL, alpha=0.35, zorder=0, label=f"Annual foreclosure rate, NHG band ({flo:g}–{fhi:g}%)")
    ax1.axhspan(dlo, dhi, color=DIS, alpha=0.30, zorder=0,
                label=f"Households in financial distress, BKR band ({dlo:g}–{dhi:g}%)")
    ax1.scatter(s.rate_tot, s.arrears_fin, s=30, color=GREY, edgecolor="none", zorder=1, label=f"Reference (n={len(s):,})")
    for tol in sorted(ABC_TOLERANCES, reverse=True):
        a = accepted[tol]
        ax1.scatter(s.loc[a, "rate_tot"], s.loc[a, "arrears_fin"], s=70, color=TOL_COLORS[tol], edgecolor="white",
                    linewidth=0.8, zorder=2, label=f"Closest {tol:.0%} (n={len(a)})")
    ax1.scatter(s.loc[band, "rate_tot"], s.loc[band, "arrears_fin"], s=600, marker="*", color=BAND,
                edgecolor="white", linewidth=0.8, zorder=3, label=f"Inside both bands (n={len(band)})")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_box_aspect(1)
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for axis, ticks in [(ax1.xaxis, [0.3, 0.5, 1, 2, 5]), (ax1.yaxis, [0.5, 1, 2, 5, 10])]:
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(NullFormatter())
    ax1.set_xlabel(r"Annual foreclosure rate at $\mathcal{S}_0$ (%)", fontsize=FS, fontweight="bold")
    ax1.set_ylabel(r"Households in financial distress at $\mathcal{S}_0$ (%)", fontsize=FS, fontweight="bold")
    ax1.legend(fontsize=FS - 6, framealpha=0, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    style(ax1)

    # right: spread of the retained combinations per parameter
    params = ABC_PARAMS[::-1]
    st = ks[ks.tol == TOL_MAIN].set_index("param")
    assert len(band) == 1, f"expected one combination inside both bands, got {len(band)}"
    star = band[0]
    bounds = load_problem()
    offsets = {0.05: 0.22, 0.01: -0.22}
    for y, p in enumerate(params):
        ax2.plot([0, 1], [y] * 2, color="black", lw=1.2, zorder=1)
        lo, hi = bounds[p]
        ax2.text(-0.015, y, fmt_bound(lo), fontsize=FS - 8, ha="right", va="center")
        ax2.text(1.015, y, fmt_bound(hi), fontsize=FS - 8, ha="left", va="center")
        for tol in ABC_TOLERANCES:
            v = pos.loc[accepted[tol], p]
            ax2.plot(v.quantile([0.05, 0.95]), [y + offsets[tol]] * 2, color=TOL_COLORS[tol], lw=7,
                     solid_capstyle="butt", zorder=3)
        if st.loc[p, "constrained"]:
            x_cal, cal = pos.loc[star, p], s.loc[star, p]
            ax2.plot(x_cal, y, marker="*", ms=30, color=BAND, markeredgecolor="white", mew=0.8, zorder=4)
            right = x_cal > 0.8                    # near the upper bound: label on the left
            ax2.text(x_cal - 0.03 if right else x_cal + 0.03, y, fmt_bound(cal), fontsize=FS - 6, fontweight="bold",
                     ha="right" if right else "left", va="center", zorder=4,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax2.set_yticks(range(len(params)))
    ax2.set_yticklabels([f"${GREEK[p]}$" + ("*" if st.loc[p, "constrained"] else "") for p in params], fontsize=FS)
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-0.6, len(params) - 0.45)
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xticklabels(["lower\nbound", "mid", "upper\nbound"])
    ax2.tick_params(axis="x", labelsize=FS - 4)
    ax2.tick_params(axis="y", length=0, pad=95)
    ax2.set_xlabel("Position in SA range", fontsize=FS, fontweight="bold")
    for sp in ["top", "right", "left"]:
        ax2.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], color=TOL_COLORS[t], lw=7, label=f"Closest {t:.0%}, 5–95%")
               for t in sorted(ABC_TOLERANCES, reverse=True)]
    handles += [plt.Line2D([], [], color="black", lw=1.2, label="SA range (bounds at the ends)")]
    handles += [plt.Line2D([], [], ls="", marker="*", ms=30, color=BAND, markeredgecolor="white", mew=0.8,
                           label="Calibrated value (inside both bands)")]
    handles += [plt.Line2D([], [], ls="", label=f"* Constrained at the {TOL_MAIN:.0%} tolerance")]
    ax2.legend(handles=handles, fontsize=FS - 7, framealpha=0, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    save_figure(fig, "figure_2", dpi=200)


if __name__ == "__main__":
    main()
