"""Download the CBS StatLine tables used to initialise the model (baseline parametrisation).

    85887NED  income and consumption expenditure by income decile  -> m, varepsilon
    80590NED  labour force and unemployment, monthly               -> eta

Each table is saved as cbs/<table>_<pull date>.csv, with the dimension codes replaced by
their labels (the same layout as the cbsodata package). CBS revises its tables, so every
pull is kept under its own date; baseline_parameters.py states which pull the paper uses.

Run:    uv run python preparation/initialization/download_cbs.py
Needs:  an internet connection; Python standard library and pandas only.
"""
import datetime
import json
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cbs")
API = "https://opendata.cbs.nl/ODataApi/odata"
TABLES = ["85887NED", "80590NED"]


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def get_all_rows(url):
    """Follow the API's paging links until every row is read."""
    rows = []
    while url:
        page = get_json(url)
        rows += page["value"]
        url = page.get("odata.nextLink")
    return rows


def download(table):
    """One table, with each dimension code replaced by its label."""
    data = pd.DataFrame(get_all_rows(f"{API}/{table}/TypedDataSet?$format=json"))
    properties = get_json(f"{API}/{table}/DataProperties?$format=json")["value"]
    dimensions = [p["Key"] for p in properties if p["Type"] in ("Dimension", "TimeDimension")]
    for dimension in dimensions:
        labels = get_json(f"{API}/{table}/{dimension}?$format=json")["value"]
        data[dimension] = data[dimension].map({d["Key"]: d["Title"] for d in labels})
    return data


def main():
    os.makedirs(OUT, exist_ok=True)
    today = datetime.date.today().isoformat()
    for table in TABLES:
        data = download(table)
        path = os.path.join(OUT, f"{table}_{today}.csv")
        data.to_csv(path, index=False)
        print(f"{table}: {len(data)} rows -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
