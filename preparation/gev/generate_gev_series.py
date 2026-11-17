"""Generate the monthly flood-severity series of the flood scenarios.

Each scenario S1..S4 is a GEV distribution (gev_scenario_parameters.py); S0 has no floods
(severity 0 in every month). The series has one column per scenario (set0..set4) and one row
per month; month t of a simulation reads row t, so every run of a scenario sees the same
flood history.

The series is drawn with NumPy's MT19937 generator, seeded with SEED (as SeedSequence
entropy). The generator is first advanced by 42,000 uniform draws, then the 2,500 months of
S1, S2, S3 and S4 are drawn in that order. The result is gev_config_static_2500.csv, the series
read by every simulation of the paper; the script checks that the two agree.

Run:    uv run python preparation/gev/generate_gev_series.py
Writes: output/gev/gev_config_static_2500.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import genextreme

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REFERENCE = os.path.join(ROOT, "preparation", "gev", "gev_config_static_2500.csv")
OUT = os.path.join(ROOT, "output", "gev", "gev_config_static_2500.csv")

SEED = 282058954875689830532290507315841628897
ADVANCE = 42_000          # uniform draws before the series
MONTHS = 2500

# (location mu, scale sigma, shape xi) of S1..S4, as in gev_scenario_parameters.py.
# scipy's genextreme uses c = -xi.
SCENARIOS = {
    1: (10.0, 27.0, 0.1),
    2: (15.789, 28.579, 0.5),
    3: (21.579, 35.421, 0.8),
    4: (28.421, 45.421, 1.2),
}


def main():
    rng = np.random.RandomState(np.random.MT19937(SEED))
    rng.random_sample(ADVANCE)
    series = {"set0": np.zeros(MONTHS)}
    for s, (mu, sigma, xi) in SCENARIOS.items():
        series[f"set{s}"] = genextreme.rvs(c=-xi, loc=mu, scale=sigma, size=MONTHS, random_state=rng)
    series = pd.DataFrame(series)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    series.to_csv(OUT, index=False)
    print(f"{MONTHS} months x {series.shape[1]} scenarios -> {OUT}")

    ref = pd.read_csv(REFERENCE, float_precision="round_trip")
    same = series.to_numpy() == ref.to_numpy()
    print(f"identical to {os.path.relpath(REFERENCE, ROOT)}: {same.sum()} of {same.size} values")
    if not same.all():
        rel = np.abs(series.to_numpy() - ref.to_numpy()) / np.maximum(np.abs(ref.to_numpy()), 1e-300)
        print(f"largest relative difference {rel.max():.1e} (scipy versions other than the locked "
              "1.11.4 round genextreme.ppf differently in the last bits)")


if __name__ == "__main__":
    main()
