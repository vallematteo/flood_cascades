# Empirical calibration targets

The model is calibrated at the no-flood setting against two Dutch credit outcomes. This
folder holds every source figure behind the two bands, where each figure is printed, and
the script that turns them into the bands.

| target | definition | years | band |
|---|---|---|---|
| foreclosure rate | NHG loss claims received / NHG active guarantees at 31 December | 2013-2016 | 0.27-0.42 % per year |
| financial distress | BKR mortgages in arrears (mid-year) / CBS households with a mortgage (1 January) | 2014-2025 | 0.87-3.54 % |

The band is the minimum and maximum of the yearly rates, rounded to two decimals.

## Reproduce

```bash
python preparation/calibration_targets/build_calibration_targets.py                                   # yearly rates and bands
python preparation/calibration_targets/build_calibration_targets.py --verify-pdfs ./source_pdfs       # download the PDFs, check every quote
python preparation/calibration_targets/build_calibration_targets.py --refresh-cbs                     # compare with the live CBS table
```

Requirements: Python 3 with pandas; `pdftotext` (poppler) for `--verify-pdfs`.

## Files

Each row of the two target files is one figure read from one document. Columns:

| column | content |
|---|---|
| `quantity`, `year`, `value`, `unit` | the figure |
| `used_for_band` | `gross` / `net` / `denominator` / `numerator` / `alternative`, or empty if not used |
| `document`, `url` | source document and download link |
| `pdf_page` | page of the PDF file, counted from 1 (not the printed page number) |
| `location` | section, table or chart, row and column |
| `verify_text` | a string that appears verbatim on that page; checked by `--verify-pdfs` |
| `note` | caveats |

- `target_foreclosure_nhg.csv`: NHG loss claims, net households claiming, dual claims and active guarantees.
- `target_distress_bkr.csv`: BKR mortgages and persons in arrears.
- `cbs_83834NED_mortgaged_households.csv`: CBS denominator, one block per pull date (`vintage`).

## Foreclosure rate (NHG)

**Numerator.** Loss claims (*verliesdeclaraties*) received by Stichting Waarborgfonds Eigen
Woningen (WEW), the fund behind the Nationale Hypotheek Garantie, in the calendar year. A
claim follows a forced sale whose proceeds do not repay the guaranteed mortgage. Each year
is read from that year's annual report (*jaarverslag*), section "Beroep op het fonds" /
"Aanspraken op de borgstelling". The 2013 figure is the comparative value in brackets in
the 2014 report.

**Denominator.** Number of active guarantees (*aantal actieve garanties*) at 31 December,
from the fund table on page 2 of NHG's quarterly reports. The annual reports do not state
this stock. 2014 and 2015 are audited annual-account values; **2016 is NHG's own estimate**
(column "Estimate at 31-12-2016" of the Q4 2016 report).

**Processing.** `rate = 100 * claims_received / active_guarantees`, per year.

| year | claims received | households claiming | active guarantees | gross rate | net rate |
|---|---|---|---|---|---|
| 2013 | 4,580 | 4,521 | 1,093,000 | 0.419 % | 0.414 % |
| 2014 | 4,864 | 4,799 | 1,175,000 | 0.414 % | 0.408 % |
| 2015 | 4,557 | 4,477 | 1,254,000 | 0.363 % | 0.357 % |
| 2016 | 3,568 | 3,504 | 1,309,000 | 0.273 % | 0.268 % |

Band from gross claims: **0.27-0.42 %**. Band from net households: 0.27-0.41 %.

**Caveats.**
- Gross claims double count losses on which both the lender and SVn (for a *starterslening*)
  filed a claim (59, 65, 80 and 64 cases). Each report gives the net number of households,
  "per saldo ... huishoudens", which equals claims minus dual claims; the script checks this.
- Active guarantees count guaranteed loans, not households.
- The series covers forced sales at a loss on NHG mortgages only (about 1.3 million of
  about 3.8 million mortgaged households), so it is a lower bound on foreclosures.
- 2017 is not used: the matching year-end stock of guarantees was not located.

**Sources.** Stichting Waarborgfonds Eigen Woningen:
- Jaarverslag 2014 (2015), p. 25, https://www.nhg.nl/media/xv1bndr5/jaarverslag_2014.pdf
- Jaarverslag 2015 (2016), p. 11, https://www.nhg.nl/media/3ldnxy3s/nhg-jv2015_def_interactive.pdf
- Jaarverslag 2016 (2017), p. 13, https://www.nhg.nl/media/bhplbg1d/wew-jaarverslag-2016.pdf
- Kwartaalcijfers 3e kwartaal 2014, p. 2, https://www.nhg.nl/media/o23fl2d4/kwartaalbericht_3e_kwartaal_2014.pdf
- Kwartaalcijfers 1e kwartaal 2016, p. 2, https://www.nhg.nl/media/nf0fcpkn/kwartaalbericht_1e_kwartaal_2016.pdf
- Quarterly Figures, Fourth Quarter 2016 (2017), p. 2, https://www.nhg.nl/media/jrpc2cde/nhg-quarterly-report-q4-2016_eng.pdf

## Financial distress (BKR / CBS)

**Numerator.** Number of mortgage contracts with a registered payment arrear, in thousands,
from the chart "Aantal hypotheken met betalingsachterstand" in section 1.2 of the BKR
Hypotheekbarometer 2025 (PDF page 5). The stock is measured mid-year (June). The values are
the chart's data labels; the chart is an image, so only 2015 (128,400) and 2025 (33,500),
which are also stated in the text below the chart, can be checked automatically. The
remaining years must be checked by eye on page 5. A BKR arrear is registered after about
three missed monthly payments.

**Denominator.** CBS StatLine table 83834NED, *Vermogen van huishoudens*: private
households (`KenmerkenVanHuishoudens = 1050010`, Particuliere huishoudens) holding the
component "2.1 Hypotheekschuld eigen woning" (`Vermogensbestanddelen = 1021461`), column
`ParticuliereHuishoudens_1`, thousands of households on 1 January of the year. API query:

```
https://opendata.cbs.nl/ODataApi/odata/83834NED/TypedDataSet?$format=json
  &$filter=KenmerkenVanHuishoudens eq '1050010' and Vermogensbestanddelen eq '1021461'
  &$select=Perioden,ParticuliereHuishoudens_1
```

**Processing.** `share = 100 * mortgages_in_arrears / households_mortgage`, with BKR year t
divided by CBS households on 1 January of t. If CBS has not yet published year t, the
last published year is used (column `denominator_year` in the script output).

The paper's band uses the CBS pull of 2026-10-09, which ends in 2024, so 2025 is divided by
the 2024 value:

- maximum: 2015, 128.4 / 3,626.1 = **3.54 %**
- minimum: 2025, 33.5 / 3,864.5 (2024 denominator) = **0.87 %**

**Caveats.**
- CBS revises the table. In the pull of 2026-11-06, 2024 is 3,848.5 and 2025 (3,904.1) is
  available, which gives a minimum of 33.5 / 3,904.1 = 0.86 %. Both pulls are stored, and
  `--cbs-vintage` selects one.
- The numerator counts mortgage contracts and the denominator counts households; BKR does
  not publish a household count. Persons in arrears (lower chart, same page) give
  0.78-3.09 %.

**Sources.**
- Stichting Bureau Krediet Registratie (2025). *Hypotheekbarometer 2025*, p. 5,
  https://www.bkr.nl/media/t0rmhyzw/bkr-hypotheekbarometer-2025.pdf
- Centraal Bureau voor de Statistiek. *Vermogen van huishoudens; huishoudenskenmerken,
  vermogensbestanddelen*, StatLine table 83834NED,
  https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83834NED

All quoted text was last checked against the downloaded PDFs on 2026-11-06.
