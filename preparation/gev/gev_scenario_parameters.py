"""
Flood severity scenarios S1..S4: how the GEV parameters are built.

The shape xi is chosen by hand (heavier tail from S1 to S4). For S2..S4 the location
mu and scale sigma are then picked so that the peak of the density sits where the
peak of GEV(10, 27, 0.2) sits. S1 keeps mu = 10, sigma = 27 with xi = 0.1.

xi > 0 is the heavy tailed case. scipy uses c = -xi.

Run:  python preparation/gev/gev_scenario_parameters.py
"""
import numpy as np
from scipy.stats import genextreme

REFERENCE = (10.0, 27.0, 0.2)        # (mu, sigma, xi), the peak every scenario is matched to
S1 = (10.0, 27.0, 0.1)
XI_LIST = [0.5, 0.8, 1.2]            # shapes of S2, S3, S4
THRESHOLDS = [0.0, 30.0, 300.0, 3000.0]
STEPS_PER_YEAR = 12


def peak(mu, sigma, xi):
    """Position and height of the density peak."""
    position = mu + sigma * ((1 + xi) ** (-xi) - 1) / xi
    height = (1 + xi) ** (1 + xi) * np.exp(-(1 + xi)) / sigma
    return position, height


def match_peak(xi, shift):
    """Grid search for the (mu, sigma) whose peak is closest to the reference peak."""
    ref_position, ref_height = peak(*REFERENCE)
    mu_grid = np.linspace(shift, 20 + shift, 20)
    sigma_grid = np.linspace(17 + shift, 37 + shift, 20)
    best = None
    for sigma in sigma_grid:
        for mu in mu_grid:
            position, height = peak(mu, sigma, xi)
            distance = ((position - ref_position) / ref_position) ** 2 + ((height - ref_height) / ref_height) ** 2
            if best is None or distance < best[0]:
                best = (distance, mu, sigma)
    return best[1], best[2], xi


def return_period(x, mu, sigma, xi):
    """Return period of severity x, in years (one draw per monthly step)."""
    return 1 / (1 - genextreme.cdf(x, c=-xi, loc=mu, scale=sigma) ** STEPS_PER_YEAR)


if __name__ == "__main__":
    # the search window moves up by 10 for each successive scenario
    scenarios = [S1] + [match_peak(xi, shift=10 * i) for i, xi in enumerate(XI_LIST)]

    print("Parameters and density peak")
    for i, (mu, sigma, xi) in enumerate(scenarios, start=1):
        position, height = peak(mu, sigma, xi)
        print(f"  S{i}: mu={mu:7.3f}  sigma={sigma:7.3f}  xi={xi:.1f}   peak at {position:5.2f}, height {height:.4f}")

    print("\nProbability per step (%): no flood, minor, moderate, severe, catastrophic")
    for i, (mu, sigma, xi) in enumerate(scenarios, start=1):
        cdf = genextreme.cdf(THRESHOLDS, c=-xi, loc=mu, scale=sigma)
        probs = 100 * np.diff(np.concatenate([[0], cdf, [1]]))
        print(f"  S{i}: " + "  ".join(f"{p:6.2f}" for p in probs))

    print("\nReturn period (years) under S1..S4 of the 10, 50 and 100 year events of S1")
    for years in (10, 50, 100):
        mu, sigma, xi = S1
        level = genextreme.ppf((1 - 1 / years) ** (1 / STEPS_PER_YEAR), c=-xi, loc=mu, scale=sigma)
        periods = "  ".join(f"{return_period(level, *s):6.2f}" for s in scenarios)
        print(f"  {years:3d}-year event (sev = {level:3.0f}): {periods}")
