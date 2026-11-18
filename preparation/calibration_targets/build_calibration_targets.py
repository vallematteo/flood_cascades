"""Rebuild the two empirical calibration bands from their source figures.

Targets (see README.md in this folder):
    foreclosure_rate    NHG loss claims / NHG active guarantees, 2013-2016   (paper: 0.27-0.42 %)
    financial_distress  BKR mortgages in arrears / CBS mortgaged households,
                        2014-2025                                            (paper: 0.87-3.54 %)

Inputs (all in this folder):
    target_foreclosure_nhg.csv           one row per figure read from an NHG/WEW report
    target_distress_bkr.csv              one row per figure read from the BKR Hypotheekbarometer
    cbs_83834NED_mortgaged_households.csv  CBS denominator, one block of rows per pull date (vintage)

Usage:
    python preparation/calibration_targets/build_calibration_targets.py                      # bands + per-year tables
    python preparation/calibration_targets/build_calibration_targets.py --verify-pdfs DIR    # download the PDFs into DIR and check
                                                             # every verify_text on its stated page
    python preparation/calibration_targets/build_calibration_targets.py --refresh-cbs        # compare the stored CBS vintage
                                                             # with the current StatLine API

--verify-pdfs needs `pdftotext` (poppler) on the PATH. Values read from chart labels have no
verify_text (the chart is an image) and must be checked by eye on the stated page.
"""
import argparse
import json
import os
import subprocess
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NHG_CSV = os.path.join(HERE, "target_foreclosure_nhg.csv")
BKR_CSV = os.path.join(HERE, "target_distress_bkr.csv")
CBS_CSV = os.path.join(HERE, "cbs_83834NED_mortgaged_households.csv")

PAPER_BANDS = {"foreclosure_rate": (0.27, 0.42), "financial_distress": (0.87, 3.54)}
PAPER_CBS_VINTAGE = "2026-10-09"

CBS_API = "https://opendata.cbs.nl/ODataApi/odata/83834NED/TypedDataSet"
CBS_FILTER = "KenmerkenVanHuishoudens eq '1050010' and Vermogensbestanddelen eq '1021461'"


def nhg_rates(nhg):
    w = nhg.pivot(index="year", columns="quantity", values="value")
    w["rate_gross"] = 100 * w.claims_received / w.active_guarantees
    w["rate_net"] = 100 * w.households_claiming / w.active_guarantees
    assert (w.claims_received - w.dual_claims == w.households_claiming).all(), \
        "households_claiming should equal claims_received - dual_claims"
    return w


def bkr_rates(bkr, cbs, vintage):
    num = bkr.pivot(index="year", columns="quantity", values="value").reset_index()
    den = (cbs[cbs.vintage == vintage][["year", "households_mortgage"]]
           .rename(columns={"year": "denominator_year"}).sort_values("denominator_year"))
    # BKR year t (mid-year stock) over CBS households on 1 January of t; when CBS has no
    # year t yet, the last available year is used (merge_asof, backward).
    t = pd.merge_asof(num.sort_values("year"), den, left_on="year", right_on="denominator_year")
    t["share_mortgages"] = 100 * t.mortgages_in_arrears / t.households_mortgage
    t["share_persons"] = 100 * t.persons_in_arrears / t.households_mortgage
    return t.set_index("year")


def band(s):
    return round(s.min(), 2), round(s.max(), 2)


def pdf_pages(pdf):
    """Text of each page, whitespace collapsed, in both reading-order and layout mode."""
    out = []
    for mode in ([], ["-layout"]):
        txt = subprocess.run(["pdftotext", *mode, pdf, "-"], capture_output=True, text=True,
                             check=True).stdout
        out.append([" ".join(p.split()) for p in txt.split("\f")])
    return list(zip(*out))


def verify_pdfs(folder, frames):
    os.makedirs(folder, exist_ok=True)
    rows = pd.concat(frames)
    cache, n_ok, n_bad = {}, 0, 0
    for url in rows.url.unique():
        path = os.path.join(folder, os.path.basename(urllib.parse.urlparse(url).path))
        if not os.path.exists(path):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                f.write(r.read())
        cache[url] = pdf_pages(path)
    for _, r in rows.iterrows():
        if not isinstance(r.verify_text, str) or not r.verify_text.strip():
            continue
        needle = " ".join(r.verify_text.split())
        found = any(needle in p for p in cache[r.url][int(r.pdf_page) - 1])
        n_ok, n_bad = n_ok + found, n_bad + (not found)
        print(f"{'OK  ' if found else 'MISS'} {r.document}, p.{int(r.pdf_page)}: "
              f"{r.quantity} {r.year} = {r.value}  ['{needle}']")
    print(f"\n{n_ok} quotes found on their stated page, {n_bad} missing")
    return n_bad == 0


def refresh_cbs(cbs):
    q = urllib.parse.urlencode({"$format": "json", "$filter": CBS_FILTER,
                                "$select": "Perioden,ParticuliereHuishoudens_1"})
    with urllib.request.urlopen(f"{CBS_API}?{q}", timeout=60) as r:
        rows = json.load(r)["value"]
    now = pd.DataFrame({"year": [int(x["Perioden"][:4]) for x in rows],
                        "current": [x["ParticuliereHuishoudens_1"] for x in rows]})
    latest = cbs[cbs.vintage == cbs.vintage.max()][["year", "households_mortgage"]]
    cmp = now.merge(latest, on="year", how="outer").query("year >= 2014")
    cmp["changed"] = cmp.current != cmp.households_mortgage
    print(f"CBS 83834NED now vs stored vintage {cbs.vintage.max()}:")
    print(cmp.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-pdfs", metavar="DIR")
    ap.add_argument("--refresh-cbs", action="store_true")
    ap.add_argument("--cbs-vintage", default=PAPER_CBS_VINTAGE)
    a = ap.parse_args()

    nhg, bkr, cbs = pd.read_csv(NHG_CSV), pd.read_csv(BKR_CSV), pd.read_csv(CBS_CSV)

    w = nhg_rates(nhg)
    print("NHG foreclosure rate, % of active guarantees per year")
    print(w[["claims_received", "households_claiming", "active_guarantees",
             "rate_gross", "rate_net"]].round(3).to_string())
    print(f"band, gross claims (paper):  {band(w.rate_gross)}")
    print(f"band, net households:        {band(w.rate_net)}\n")

    for v in sorted(cbs.vintage.unique()):
        t = bkr_rates(bkr, cbs, v)
        print(f"BKR financial distress, % of CBS mortgaged households (CBS vintage {v})")
        print(t[["mortgages_in_arrears", "persons_in_arrears", "denominator_year",
                 "households_mortgage", "share_mortgages", "share_persons"]].round(3).to_string())
        print(f"band, mortgages in arrears:  {band(t.share_mortgages)}"
              f"{'  (paper)' if v == a.cbs_vintage else ''}")
        print(f"band, persons in arrears:    {band(t.share_persons)}\n")

    got = {"foreclosure_rate": band(w.rate_gross),
           "financial_distress": band(bkr_rates(bkr, cbs, a.cbs_vintage).share_mortgages)}
    for k, v in PAPER_BANDS.items():
        print(f"{k}: rebuilt {got[k]}, paper {v} -> {'MATCH' if got[k] == v else 'DIFFERENT'}")

    if a.refresh_cbs:
        print()
        refresh_cbs(cbs)
    if a.verify_pdfs:
        print()
        verify_pdfs(a.verify_pdfs, [nhg, bkr])


if __name__ == "__main__":
    main()
