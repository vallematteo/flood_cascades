"""SI Fig. 7: sensitivity of the mean number of distress months per household across the five
flood scenarios.

(a) first-order Sobol' indices (Saltelli design of 3,328 vectors, 95% bootstrap intervals);
(b) median PAWN indices with S = 10 conditioning intervals, bars from the minimum to the maximum
over the intervals, dotted line at the 95% max-T threshold of 10,000 permutations.

Run:    uv run python figure_S7.py
Writes: output/figures/figure_S7a.png, figure_S7b.png
"""
from utils import pawn_figure, sobol_figure

if __name__ == "__main__":
    sobol_figure("mean_months", "S1", "figure_S7a")
    pawn_figure("mean_months", "figure_S7b")
