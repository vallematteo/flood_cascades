"""SI Fig. 2: model geography and the spatial distribution of foreclosures in Limburg.

(a) relative housing density; (b) nested flood extents of the four national flood maps, with a
magnified inset; (c) local annual foreclosure rate at S0; (d) log2 ratio of the local rate at S4
to S0, with the outer flood extent. The local rate is the calibration's foreclosure rate per
cell: per run, foreclosures are smoothed with a Gaussian kernel (bandwidth 10 cells) and divided
by the run's mean active mortgages times the cell's share of the smoothed housing density; maps
are means over runs. Runs: cascade experiment, delta = 0, kappa = 10, beta1 - beta2 = 0.5,
vartheta = 0.05; S0 20 runs, S4 60 runs (lambda in {0.25, 0.5, 0.75}).

Run:    uv run python figure_S2.py
Writes: output/figures/figure_S2.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter

from utils import FS, PREPARATION, load_cascade_runs, read, save_figure

MAPS = os.path.join(PREPARATION, "flood_maps")
plt.rcParams.update({"font.size": FS, "axes.labelweight": "bold",
                     "axes.titleweight": "bold", "axes.titlesize": FS})

FLOOD_SETTING = 4        # scenario compared with S0 in panel (d)
BANDWIDTH = 10.0         # kernel standard deviation, in grid cells
MIN_POP = 0.05           # mask the ratio where smoothed housing < this fraction of its maximum
INSET = (362, 477, 112, 227)   # row0, row1, col0, col1 of the inset of panel (b)

plt.rcParams.update({"font.size": FS, "axes.labelweight": "bold",
                     "axes.titleweight": "bold", "axes.titlesize": FS})

# Smallest to largest extent; tier names as in SI Table flood_severity.
TIERS = [("Minor", "grote.png"), ("Moderate", "middel.png"),
         ("Severe", "kleine.png"), ("Catastrophic", "extreme_kleine.png")]

REGION_FILL = "#ececec"
REGION_EDGE = "#8c8c8c"
# Band colours: index 1 (catastrophic only) light ... 4 (all tiers) dark. Plasma samples
# (multi-hue, so neighbouring steps differ in hue as well as lightness), with no plain blue, so the
# bands are not read as the scenario palette (S1-S4 are Blues in the paper figures).
BAND_COLOURS = ["#f9dc24", "#f2844b", "#b6308b", "#2a0593"]

_reds = LinearSegmentedColormap.from_list("seq_red", ["#fee0d2", "#fcbba1", "#fb6a4a",
                                                      "#cb181d", "#67000d"])(np.linspace(0, 1, 256))
_reds[:, 3] = np.clip(np.linspace(0, 1, 256) / 0.15, 0, 1)  # low end fades out
SEQ_RED = ListedColormap(_reds, name="seq_red_fade")
SEQ_GREY = LinearSegmentedColormap.from_list("seq_grey", ["#f0f0f0", "#969696", "#252525"])


def read_map(name):
    """Same transform as read_map in model/utils.jl: invert grey, scale to [0, 1]."""
    a = np.asarray(Image.open(os.path.join(MAPS, name)).convert("L"), dtype=float) / 255
    a = 1.0 - a
    return a / a.max() if a.max() > 0 else a


def flood_bands():
    """Number of tiers flooding each cell (0..4), after checking the extents are nested."""
    ext = [read_map(f) > 0 for _, f in TIERS]
    for (n_small, _), (n_big, _), a, b in zip(TIERS, TIERS[1:], ext, ext[1:]):
        bad = int((a & ~b).sum())
        if bad:
            raise SystemExit(f"Extents not nested: {bad} {n_small} cells outside {n_big}")
    return sum(e.astype(int) for e in ext)


def kde_grid(rows, cols, shape, bw, weights=None):
    """Gaussian KDE on the grid: point histogram convolved with N(0, bw^2 I), reflecting
    boundary, so the estimate integrates to the number (or weight) of points."""
    h = np.zeros(shape)
    np.add.at(h, (rows, cols), 1.0 if weights is None else weights)
    return gaussian_filter(h, sigma=bw, mode="reflect", truncate=4.0)


def rate_map(runs, fcl, shape, pop_share, bw):
    """Local annual foreclosure rate (%), defined as the calibration's rate_tot
    (SA_4/analysis/distress_index_4.py): per run, 100 * foreclosures / mean_active / years,
    then the mean over seeds. Locally, foreclosures are the run's KDE and mean_active is
    the run's mean active population times the cell's share of the smoothed placement
    weight. The mean over runs is one KDE with each point weighted by
    100 / (n_runs * mean_active_run * years_run)."""
    w_run = 100 / (len(runs) * runs.mean_active * (runs.tot_step - runs.warmup) / 12)
    w = fcl.run_id.map(w_run.set_axis(runs.run_id)).to_numpy()
    num = kde_grid(fcl.y.values - 1, fcl.x.values - 1, shape, bw, weights=w)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = num / pop_share
    # Check: averaging the map over the population gives back the seed-mean rate_tot.
    print(f"  population-weighted mean of map {np.nansum(rate * pop_share):.3f} %/yr "
          f"vs seed-mean rate_tot {runs.rate_tot.mean():.3f} %/yr")
    return rate


def flat(colour):
    return LinearSegmentedColormap.from_list("c", [colour, colour])


def base(ax, region):
    ax.imshow(np.where(region, 1.0, np.nan), cmap=flat(REGION_FILL), interpolation="nearest")


def outline(ax, region):
    ax.contour(region, levels=[0.5], colors=[REGION_EDGE], linewidths=1.0)


def finish(ax, letter, title):
    ax.set_axis_off()
    ax.set_title(f"({letter}) {title}", fontsize=FS - 2, pad=10, loc="left")


def hcbar(fig, im, cax, label, **kw):
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", **kw)
    cb.set_label(label, fontsize=FS - 6)
    cb.ax.tick_params(labelsize=FS - 9)
    cb.outline.set_visible(False)
    return cb


def draw_bands(ax, band, region):
    cmap = ListedColormap(BAND_COLOURS)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    base(ax, region)
    ax.imshow(np.ma.masked_where(band == 0, band), cmap=cmap, norm=norm, interpolation="nearest")
    outline(ax, region)


def plot(housing, region, band, rate0, rate_vmax, ratio, inset_box):
    # Columns: (a) | (b) map | (b) magnified inset | (c) | (d)
    fig = plt.figure(figsize=(33, 17))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.022], width_ratios=[1, 1, 1.25, 1, 1],
                          hspace=0.06, wspace=0.05)
    ax_a, ax_b, axin, ax_c, ax_d = [fig.add_subplot(gs[0, i]) for i in range(5)]
    caxes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:3]),
             fig.add_subplot(gs[1, 3]), fig.add_subplot(gs[1, 4])]
    for k, cax in enumerate(caxes):  # colour bars span the middle of their slot
        p = cax.get_position()
        f = 0.1 if k != 1 else 0.2
        cax.set_position([p.x0 + f * p.width, p.y0, (1 - 2 * f) * p.width, p.height])

    # (a) housing density
    base(ax_a, region)
    im = ax_a.imshow(np.ma.masked_where(~region | (housing <= 0), housing), cmap=SEQ_GREY,
                     vmin=0, vmax=1, interpolation="nearest")
    outline(ax_a, region)
    finish(ax_a, "a", "Housing density")
    hcbar(fig, im, caxes[0], "Relative housing density")

    # (b) nested flood extents, with a magnified inset in its own column
    draw_bands(ax_b, band, region)
    finish(ax_b, "b", "Flood extent")
    r0, r1, c0, c1 = inset_box
    ax_b.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="#404040", lw=1.5))
    draw_bands(axin, band, region)
    axin.set_xlim(c0, c1)
    axin.set_ylim(r1, r0)
    axin.set_xticks([]), axin.set_yticks([])
    for sp in axin.spines.values():
        sp.set_edgecolor("#404040"), sp.set_linewidth(1.5)
    fig.canvas.draw()  # fix the inset's final position before drawing the connectors
    for (xb, yb), (xi, yi) in [((c1, r0), (c0, r0)), ((c1, r1), (c0, r1))]:
        fig.add_artist(ConnectionPatch(xyA=(xb, yb), coordsA=ax_b.transData, xyB=(xi, yi),
                                       coordsB=axin.transData, color="#404040", lw=1.2))
    # Swatch strip, dark to light, labelled by the probability class of the national flood
    # maps (grote, middelgrote, kleine, extreem kleine kans; main text: "frequent to extremely
    # rare"). The extents are nested, so each cell is labelled by the most frequent map that
    # covers it, and it floods in that event and in every rarer one.
    cax = caxes[1]
    cax.imshow(np.arange(4)[None, ::-1], cmap=ListedColormap(BAND_COLOURS), aspect="auto",
               extent=(0, 4, 0, 1))
    cax.set_xticks([0.5, 1.5, 2.5, 3.5])
    cax.set_xticklabels(["Frequent", "Medium", "Rare", "Extremely\nrare"], fontsize=FS - 9)
    cax.tick_params(length=0)
    cax.set_yticks([])
    cax.set_xlabel("Most frequent flood reaching the cell", fontsize=FS - 6, fontweight="bold")
    for sp in cax.spines.values():
        sp.set_visible(False)

    # (c) foreclosure rate at S0 (same variable as the ratio in (d))
    shown = rate0
    base(ax_c, region)
    im = ax_c.imshow(np.ma.masked_invalid(shown), cmap=SEQ_RED, vmin=0,
                     vmax=rate_vmax, interpolation="bilinear")
    outline(ax_c, region)
    finish(ax_c, "c", "Foreclosure rate, $\\mathcal{S}_0$")
    hcbar(fig, im, caxes[2], "Per 100 households per year", extend="max")

    # (d) log2 rate ratio S_flood / S0, with the outer flood extent (smoothed) on top
    lim = np.nanpercentile(np.abs(ratio), 99)
    base(ax_d, region)
    im = ax_d.imshow(np.ma.masked_invalid(ratio), cmap="RdBu_r", vmin=-lim, vmax=lim,
                     interpolation="bilinear")
    extent = gaussian_filter((band >= 1).astype(float), sigma=2) > 0.25
    ax_d.contour(extent, levels=[0.5], colors=["#252525"], linewidths=0.8)
    outline(ax_d, region)
    finish(ax_d, "d", f"Rate change, $\\mathcal{{S}}_{FLOOD_SETTING}$ vs $\\mathcal{{S}}_0$")
    hcbar(fig, im, caxes[3], "$\\log_2$ rate ratio", extend="both")

    save_figure(fig, "figure_S2", dpi=200)


def load_map_runs():
    """The runs drawn in the maps (S0 and FLOOD_SETTING) and their foreclosures per cell."""
    r = load_cascade_runs()
    r = r[r.success & (r.block == "main") & np.isclose(r.delta, 0.0) & np.isclose(r.kappa, 10.0)
          & np.isclose(r.beta2, 0.5) & np.isclose(r.vartheta, 0.05)]
    cells = read("cascades_rnet/fcl_cells.csv.gz")
    return {s: (r[r.setting == s], cells[cells.run_id.isin(r.run_id[r.setting == s])]) for s in (0, FLOOD_SETTING)}


def main():
    housing = read_map("nvm_cleaned.png")
    region = binary_fill_holes(binary_closing(housing > 0, iterations=3))
    band = np.where(region, flood_bands(), 0)

    pop_share = kde_grid(*np.nonzero(housing >= 0), housing.shape, BANDWIDTH, weights=housing[housing >= 0])
    pop_share /= pop_share.sum()

    fields = {}
    for s, (runs, cells) in load_map_runs().items():
        fcl = cells.loc[cells.index.repeat(cells.n_fcl)]          # one row per foreclosure
        fields[s] = rate_map(runs, fcl, housing.shape, pop_share, BANDWIDTH)

    keep = region & (pop_share >= MIN_POP * pop_share.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(keep, np.log2(fields[FLOOD_SETTING] / fields[0]), np.nan)
        rate0 = np.where(keep, fields[0], np.nan)
        rate_flood = np.where(keep, fields[FLOOD_SETTING], np.nan)
    ratio[~np.isfinite(ratio)] = np.nan
    # colour scale of (c): 99.5th percentile of the flooded-scenario rate
    rate_vmax = np.nanpercentile(rate_flood, 99.5)
    plot(housing, region, band, rate0, rate_vmax, ratio, INSET)


if __name__ == "__main__":
    main()
