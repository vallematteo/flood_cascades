"""SI Fig. 3: active mortgage population without floods in the (beta1 - beta2, delta) plane.

Each point is a parameter vector of the sensitivity analysis, coloured by its mean number of
active mortgages at S0 (mean over seeds of the post-warm-up mean). Background: the same
population smoothed with a Gaussian kernel (bandwidth 0.045 of each axis range); contours:
iso-population lines of a wider kernel (0.07). Dashed line: beta1 - beta2 = 1/2, the boundary
of the damage-required region.

Run:    uv run python figure_S3.py
Writes: output/figures/figure_S3.png
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from utils import FS, load_gsa_runs, save_figure

GAP = 0.5
BW = 0.045               # kernel of the background
BW_CONTOUR = 0.07        # kernel of the contours
TINT = 0.6
LEVELS = [500, 1000, 1500, 2000, 2500]
POP_MAX = 2600           # top of the colour scale
DELTA_LIM = (0.05, 0.5)  # sampling range of delta
CMAP = LinearSegmentedColormap.from_list("pop", ["#B2182B", "#FF8C00"])


def vectors():
    """Per parameter vector: beta1 - beta2, delta and the mean active mortgages at S0."""
    runs = load_gsa_runs()
    by_set = runs.groupby(["saltelli_sample_id", "setting"]).mean_active.mean().unstack()
    par = runs.groupby("saltelli_sample_id")[["beta1", "beta2", "delta"]].first()
    par["gap"] = par.beta1 - par.beta2
    par["pop0"] = by_set[0]
    return par


def smoothed_pop(par, n=200, bw=BW):
    """Gaussian-kernel smoothed S0 population on an n x n grid over [0, 1] x DELTA_LIM."""
    gx = np.linspace(0.0, 1.0, n)
    gy = np.linspace(*DELTA_LIM, n)
    bwy = bw * (DELTA_LIM[1] - DELTA_LIM[0])
    x, y, v = par.gap.values, par.delta.values, par.pop0.values
    out = np.empty((n, n))
    for j, d in enumerate(gy):
        w = np.exp(-((gx[:, None] - x) ** 2 / (2 * bw ** 2) + (d - y) ** 2 / (2 * bwy ** 2)))
        out[j] = (w @ v) / w.sum(axis=1)
    return gx, gy, out


def main():
    par = vectors()
    fig, ax = plt.subplots(figsize=(10, 9))
    gx, gy, pop = smoothed_pop(par)
    rgb = 1 - TINT * (1 - CMAP(Normalize(0, POP_MAX, clip=True)(pop))[..., :3])
    ax.imshow(rgb, origin="lower", extent=(0, 1, *DELTA_LIM), interpolation="bilinear", aspect="auto", zorder=0)

    damage_required = par[par.gap >= GAP]
    _, _, pop_c = smoothed_pop(par, bw=BW_CONTOUR)
    cs = ax.contour(gx, gy, pop_c, levels=LEVELS, colors="black", linewidths=1.2, alpha=0.6, zorder=1)
    ax.clabel(cs, fmt="%d", fontsize=FS - 9)
    sc = ax.scatter(par.gap, par.delta, c=par.pop0, cmap=CMAP, vmin=0, vmax=POP_MAX,
                    s=14, alpha=0.8, edgecolors="none", zorder=2)
    ax.axvline(GAP, color="black", lw=2.5, ls="--")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(*DELTA_LIM)
    ax.set_xlabel(r"$\beta_1-\beta_2$", fontsize=FS, fontweight="bold")
    ax.set_ylabel(r"$\delta$", fontsize=FS, fontweight="bold")
    ax.grid(linestyle="-", alpha=0.2, color="gray")
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.5)
    ax.tick_params(labelsize=FS)
    ax.grid(False)

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.solids.set_alpha(1)
    cb.set_label(r"Active mortgages at $\mathcal{S}_0$", fontsize=FS - 3, fontweight="bold")
    cb.ax.tick_params(labelsize=FS - 5)
    handles = [Line2D([], [], color="black", lw=2.5, ls="--",
                      label=rf"$\beta_1-\beta_2=1/2$ (damage-required: {len(damage_required):,})"),
               Line2D([], [], color="black", lw=1.2, alpha=0.6, label="Iso-population (smoothed)")]
    ax.legend(handles=handles, fontsize=FS - 5, framealpha=0, edgecolor="black", fancybox=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    fig.tight_layout()
    save_figure(fig, "figure_S3", dpi=300)


if __name__ == "__main__":
    main()
