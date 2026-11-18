"""Fig. 3: foreclosure rate and composition.

(a) Annual foreclosure rate by flood scenario, split into financial and strategic foreclosures,
    at the calibrated parametrisation with beta1 = 1, beta2 = 0.5, beta3 = 1, kappa = 10,
    vartheta = 0.05, delta = 0, lambda in {0.25, 0.5, 0.75} pooled: mean over runs, bar = 95%
    interval of the total rate (1.96 standard errors). Data: cascade experiment, delta = 0.
(b) Share of financial and strategic foreclosures against beta1 - beta2, in 12 quantile bins,
    for the sensitivity-analysis vectors with delta <= 0.15; lower panel: mean active mortgages.
    Data: sensitivity analysis.

Run:    python figure_3.py
Writes: output/figures/figure_3a.png, output/figures/figure_3b.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (CALIBRATED, CHANNEL_COLORS, FS, SETTING_COLORS, YEARS, load_cascade_runs,
                   load_gsa_runs, load_gsa_vectors, load_saltelli_matrix, save_figure)

DELTA_MAX = 0.15          # panel (b): vectors with a small foreclosure discount
N_BINS = 12               # panel (b): quantile bins of beta1 - beta2
MARKERS = {0: "o", 1: "s", 2: "^", 3: "D", 4: "P"}


def panel_a():
    r = load_cascade_runs()
    r = r[(r.delta == 0) & (r.kappa == 10) & (r.vartheta == 0.05) & (r.beta2 == 0.5)].copy()
    for k, v in CALIBRATED.items():
        assert np.allclose(r[k], v), k
    for ch in ("tot", "fin", "strat"):
        r[f"rate_{ch}"] = 100 * r[f"n_fcl_{ch}"] / r.mean_active / YEARS
    g = r.groupby("setting")[["rate_fin", "rate_strat", "rate_tot"]]
    mean, ci = g.mean(), 1.96 * g.std(ddof=1) / np.sqrt(g.count())
    x = mean.index.values

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x, mean.rate_fin, color=CHANNEL_COLORS["fin"], label="financial", width=0.65)
    ax.bar(x, mean.rate_strat, bottom=mean.rate_fin, color=CHANNEL_COLORS["strat"], label="strategic", width=0.65)
    ax.errorbar(x, mean.rate_tot, yerr=ci.rate_tot, fmt="none", ecolor="black", elinewidth=2, capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\mathcal{{S}}_{s}$" for s in x])
    ax.set_xlabel("Flood scenario", fontsize=FS, fontweight="bold")
    ax.set_ylabel("Foreclosure rate (%)", fontsize=FS, fontweight="bold")
    ax.grid(linestyle="-", alpha=0.2, color="gray")
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.5)
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS - 5, framealpha=0, fancybox=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    fig.tight_layout()
    save_figure(fig, "figure_3a", dpi=200)


def share_in_bin(g, col):
    """Share of all foreclosures of the bin in channel col (%), ratio estimator, with the
    delta-method 95% interval."""
    x, t = g[col].to_numpy(), g.n_total.to_numpy()
    r = x.sum() / t.sum()
    se = np.sqrt(np.var(x - r * t, ddof=1) / len(g)) / t.mean()
    return pd.Series({"gap": g.gap.mean(), "mean": 100 * r, "ci": 100 * 1.96 * se})


def panel_b():
    design = load_saltelli_matrix()[["saltelli_sample_id", "beta1", "beta2", "delta"]]
    d = load_gsa_vectors().merge(design, on="saltelli_sample_id")
    d["gap"] = d.beta1 - d.beta2
    d = d[d.delta <= DELTA_MAX].copy()
    d["gap_bin"] = pd.qcut(d.gap, N_BINS, duplicates="drop")
    active = load_gsa_runs().groupby(["setting", "saltelli_sample_id"], as_index=False).mean_active.mean()
    settings = sorted(d.setting.unique())
    channels = [("n_strategic", "Strategic", "-"), ("n_financial", "Financial", "--")]

    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for col, _, linestyle in channels:
        summary = (d.groupby(["gap_bin", "setting"], observed=True)
                   .apply(share_in_bin, col, include_groups=False).reset_index())
        for s in settings:
            z = summary[summary.setting == s].sort_values("gap")
            x, y, ci = z.gap.to_numpy(), z["mean"].to_numpy(), z.ci.to_numpy()
            ax.plot(x, y, color=SETTING_COLORS[s], linestyle=linestyle, marker=MARKERS[s], ms=5.5, lw=2.2, zorder=3)
            ax.fill_between(x, y - ci, y + ci, color=SETTING_COLORS[s], alpha=0.055, linewidth=0)

    # lower panel: mean active mortgages per bin
    dn = d[["setting", "saltelli_sample_id", "gap", "gap_bin"]].merge(active, on=["setting", "saltelli_sample_id"])
    cnt = (dn.groupby(["gap_bin", "setting"], observed=True)
           .agg(gap=("gap", "mean"), mean=("mean_active", "mean"), sd=("mean_active", "std"), n=("mean_active", "size"))
           .reset_index())
    cnt["ci"] = 1.96 * cnt.sd / np.sqrt(cnt.n)
    for s in settings:
        z = cnt[cnt.setting == s].sort_values("gap")
        x, y, ci = z.gap.to_numpy(), z["mean"].to_numpy(), z.ci.to_numpy()
        ax_n.plot(x, y, color=SETTING_COLORS[s], marker=MARKERS[s], ms=5.5, lw=2.2, zorder=3)
        ax_n.fill_between(x, np.clip(y - ci, 1e-9, None), y + ci, color=SETTING_COLORS[s], alpha=0.12, linewidth=0)
    ax_n.axhline(1000, color="black", lw=1.2, ls="--", alpha=0.5, zorder=1)
    ax_n.set_ylabel("Active households", fontsize=FS - 6, fontweight="bold")
    ax_n.set_ylim(bottom=0)
    ax_n.tick_params(axis="both", labelsize=FS - 6)
    ax_n.grid(axis="y", ls="--", alpha=0.35)
    ax_n.spines["right"].set_visible(False)
    ax_n.spines["top"].set_visible(False)

    ax_n.set_xlabel(r"$\beta_1-\beta_2$", fontsize=FS, fontweight="bold")
    ax.set_ylabel("Share of defaults (%)", fontsize=FS - 2, fontweight="bold")
    ax.set_ylim(-2, 102)
    ax.tick_params(axis="both", labelsize=FS - 6)
    ax.grid(axis="y", ls="--", alpha=0.35)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    setting_handles = [ax.plot([], [], color=SETTING_COLORS[s], marker=MARKERS[s], lw=2.2,
                               label=fr"$\mathcal{{S}}_{s}$")[0] for s in settings]
    channel_handles = [ax.plot([], [], color="#222222", linestyle=ls, lw=2.5, label=label)[0]
                       for _, label, ls in channels]
    legend_settings = ax_n.legend(handles=setting_handles, title="Flood scenario", ncol=len(settings),
                                  loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False,
                                  fontsize=FS - 4, title_fontsize=FS - 4)
    ax_n.add_artist(legend_settings)
    ax_n.legend(handles=channel_handles, title="Default type", ncol=2, loc="upper center",
                bbox_to_anchor=(0.5, -0.64), frameon=False, fontsize=FS - 4, title_fontsize=FS - 4)
    ax_n.set_xlim(left=0)
    for a in fig.axes:                     # grey band over the damage-required region beta1 - beta2 > 1/2
        xlim = a.get_xlim()
        a.axvspan(0.5, xlim[1], color="grey", alpha=0.15, linewidth=0, zorder=0)
        a.set_xlim(xlim)
    fig.tight_layout()
    save_figure(fig, "figure_3b", dpi=300)


if __name__ == "__main__":
    panel_a()
    panel_b()
