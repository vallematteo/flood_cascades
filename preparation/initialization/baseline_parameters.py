"""Baseline values of the parameters with a direct empirical counterpart (SI, Baseline parametrisation).

    m, varepsilon   non-housing expenditure share, s = m / (1 + varepsilon * I), fitted by
                    nonlinear least squares to the CBS income deciles (85887NED, 2021-2023)
    exp_lb          subsistence floor: mean Dutch social assistance norm (bijstandsnorm) 2021-2023
    eta             steady-state unemployment rate: mean monthly seasonally adjusted rate
                    2021-2025 (CBS 80590NED)

These values initialise the model and centre the parameter ranges of the global sensitivity
analysis, whose ranges are also the prior of the calibration (ABC).

Run:    uv run python preparation/initialization/baseline_parameters.py
Reads:  cbs/85887NED_2026-03-16.csv, cbs/80590NED_2026-11-17.csv (from download_cbs.py)
Writes: baseline_parameters.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
INCOME_DECILES = os.path.join(HERE, "cbs", "85887NED_2026-03-16.csv")
UNEMPLOYMENT = os.path.join(HERE, "cbs", "80590NED_2026-11-17.csv")

# Social assistance norm (bijstandsnorm), EUR per month, from the Ministry of Social Affairs
# and Employment (SZW) publications cited in the SI.
BIJSTANDSNORM = {2021: 1021, 2022: 1046, 2023: 1195}

MONTHS = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus",
          "september", "oktober", "november", "december"]


def expenditure_share_by_decile():
    """Monthly income (EUR) and non-housing expenditure share of the 30 decile-years."""
    cbs = pd.read_csv(INCOME_DECILES)
    deciles = cbs[cbs.Huishoudenskenmerken.str.contains("Beschikbaar inkomen .*10%-groep", regex=True)
                  & cbs.Perioden.astype(str).isin(["2021", "2022", "2023"])]
    income = deciles["BrutoBeschikbaarInkomen_34"]             # thousand EUR per year
    consumption = deciles["TotaalConsumptieveBestedingen_37"]
    housing = deciles["HuisvestingWaterEnEnergie_41"]          # COICOP 04
    income_monthly = (income * 1000 / 12).to_numpy()
    share = ((consumption - housing) / income).to_numpy()
    return income_monthly, share


def r_squared(observed, predicted):
    return 1 - ((observed - predicted) ** 2).sum() / ((observed - observed.mean()) ** 2).sum()


def fit_expenditure_share(income_monthly, share):
    """Nonlinear least squares in share space, income in thousand EUR for numerical stability."""
    income_k = income_monthly / 1000

    def engel(i, m, eps_k):
        return m / (1 + eps_k * i)

    estimate, covariance = curve_fit(engel, income_k, share, p0=[share.max() * 1.5, 1.0],
                                     bounds=([0, 0], [np.inf, np.inf]))
    se = np.sqrt(np.diag(covariance))
    return {
        "m": estimate[0],
        "se_m": se[0],
        "varepsilon": estimate[1] / 1000,                       # per EUR
        "se_varepsilon": se[1] / 1000,
        "corr_m_varepsilon": covariance[0, 1] / (se[0] * se[1]),
        "R2": r_squared(share, engel(income_k, *estimate)),
    }


def fit_linearised_ols(income_monthly, share):
    """Robustness check: regress 1/s on income, 1/s = 1/m + (varepsilon/m) I."""
    income_k = income_monthly / 1000
    slope, intercept = np.polyfit(income_k, 1 / share, 1)
    m, eps_k = 1 / intercept, slope / intercept
    return {"m_OLS": m, "R2_OLS": r_squared(share, m / (1 + eps_k * income_k))}


def unemployment_rate(first_year=2021, last_year=2025):
    """Mean monthly seasonally adjusted unemployment rate (%), all persons aged 15 to 75."""
    cbs = pd.read_csv(UNEMPLOYMENT)
    total = cbs[(cbs.Geslacht == "Totaal") & (cbs.Leeftijd == "15 tot 75 jaar")]
    period = total.Perioden.str.split()
    monthly = period.str.len().eq(2) & period.str[1].isin(MONTHS)
    year = period.str[0].astype(int)
    in_window = monthly & year.between(first_year, last_year)
    rates = total.loc[in_window, "Seizoengecorrigeerd_8"]       # unemployment rate, seasonally adjusted
    return {"eta_data_pct": rates.mean(), "eta_months": int(in_window.sum())}


def main():
    income_monthly, share = expenditure_share_by_decile()
    results = {"n_decile_years": len(share),
               "income_min_eur_month": income_monthly.min(),
               "income_max_eur_month": income_monthly.max()}
    results |= fit_expenditure_share(income_monthly, share)
    results |= fit_linearised_ols(income_monthly, share)
    results["exp_lb"] = np.mean(list(BIJSTANDSNORM.values()))
    results |= unemployment_rate()

    table = pd.Series(results, name="value").rename_axis("quantity").reset_index()
    table.to_csv(os.path.join(HERE, "baseline_parameters.csv"), index=False)
    with pd.option_context("display.float_format", "{:.6g}".format):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
