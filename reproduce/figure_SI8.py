"""SI Fig. 8: total-order Sobol' indices of the mean number of distress months per household
across the five flood scenarios (Saltelli design of 3,328 vectors, 95% bootstrap intervals).

Run:    python reproduce/figure_SI8.py
Writes: output/figures/figure_S8.png
"""
from utils import sobol_figure

if __name__ == "__main__":
    sobol_figure("mean_months", "ST", "figure_S8")
