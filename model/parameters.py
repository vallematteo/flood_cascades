"""Parameter values of model/parameters.csv, for the Python scripts (the Julia runners read the
same file through model/parameters.jl)."""
import os

import pandas as pd

PARAMETERS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.csv")


def model_parameters(*sets):
    """{parameter: value} of the given sets, later sets overriding earlier ones. The calibrated
    parametrisation is model_parameters("fixed", "baseline", "calibrated")."""
    table = pd.read_csv(PARAMETERS_CSV, float_precision="round_trip")   # floats read exactly
    values = {}
    for s in sets:
        rows = table[table.set == s]
        values.update(zip(rows.parameter, rows.value.astype(float)))
    return values
