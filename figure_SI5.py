"""SI Fig. 5: sensitivity of the number of financial foreclosures per run across the five flood scenarios.

(a) first-order and (b) total-order Sobol' indices (Saltelli design of 3,328 vectors, 95%
bootstrap intervals); (c) median PAWN indices with S = 10 conditioning intervals on the A and B
blocks of the design (512 vectors), bars from the minimum to the maximum over the intervals,
dotted line at the 95% max-T threshold of 10,000 permutations.

Run:    uv run python figure_SI5.py
Writes: output/figures/figure_S5a.png, figure_S5b.png, figure_S5c.png
"""
from utils import pawn_figure, sobol_figure

if __name__ == "__main__":
    sobol_figure("n_financial", "S1", "figure_S5a")
    sobol_figure("n_financial", "ST", "figure_S5b")
    pawn_figure("n_financial", "figure_S5c")
