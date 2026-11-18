"""SI Fig. 6: sensitivity of the number of strategic foreclosures per run across the five flood scenarios.

(a) first-order and (b) total-order Sobol' indices (Saltelli design of 3,328 vectors, 95%
bootstrap intervals); (c) median PAWN indices with S = 10 conditioning intervals on the A and B
blocks of the design (512 vectors), bars from the minimum to the maximum over the intervals,
dotted line at the 95% max-T threshold of 10,000 permutations.

Run:    python figure_SI6.py
Writes: output/figures/figure_S6a.png, figure_S6b.png, figure_S6c.png
"""
from utils import pawn_figure, sobol_figure

if __name__ == "__main__":
    sobol_figure("n_strategic", "S1", "figure_S6a")
    sobol_figure("n_strategic", "ST", "figure_S6b")
    pawn_figure("n_strategic", "figure_S6c")
