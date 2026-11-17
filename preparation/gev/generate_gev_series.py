"""Draw a monthly flood-severity series for each flood scenario.

Each scenario S1..S4 is a GEV distribution (gev_scenario_parameters.py); S0 has no floods
(severity 0 in every month). The series has one column per scenario (set0..set4) and one row
per month; month t of a simulation reads row t, so every run of a scenario sees the same
flood history.

The simulations of the paper use the reference series gev_config_static_2500.csv, drawn
once from these distributions; that draw cannot be regenerated (it was made from an unseeded
random state). This script draws a new series of the same form from a seeded generator, for
simulations with another flood history. It never overwrites the reference series.

Run:    uv run python preparation/gev/generate_gev_series.py [--seed 42] [--months 2500]
Writes: output/gev/gev_series_seed<seed>_<months>.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import genextreme

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# (location mu, scale sigma, shape xi) of S1..S4, as in gev_scenario_parameters.py.
# scipy's genextreme uses c = -xi.
SCENARIOS = {
    1: (10.0, 27.0, 0.1),
    2: (15.789, 28.579, 0.5),
    3: (21.579, 35.421, 0.8),
    4: (28.421, 45.421, 1.2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--months", type=int, default=2500)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    series = {"set0": np.zeros(a.months)}
    for s, (mu, sigma, xi) in SCENARIOS.items():
        series[f"set{s}"] = genextreme.rvs(c=-xi, loc=mu, scale=sigma, size=a.months, random_state=rng)

    out = os.path.join(ROOT, "output", "gev", f"gev_series_seed{a.seed}_{a.months}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(series).to_csv(out, index=False)
    print(f"{a.months} months x {len(series)} scenarios -> {out}")


if __name__ == "__main__":
    main()
