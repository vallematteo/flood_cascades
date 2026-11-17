"""SI Fig. 4: sensitivity of the share of financial foreclosures across the five flood scenarios.

(a) first-order and (b) total-order Sobol' indices (Saltelli design of 3,328 vectors, 95%
bootstrap intervals); (c) median PAWN indices with S = 10 conditioning intervals on the A and B
blocks of the design (512 vectors), bars from the minimum to the maximum over the intervals,
dotted line at the 95% max-T threshold of 10,000 permutations.

Run:    uv run python figure_S4.py
Writes: output/figures/figure_S4a.png, figure_S4b.png, figure_S4c.png
"""
from utils import pawn_figure, sobol_figure

if __name__ == "__main__":
    sobol_figure("perc_fin", "S1", "figure_S4a")
    sobol_figure("perc_fin", "ST", "figure_S4b")
    pawn_figure("perc_fin", "figure_S4c")
