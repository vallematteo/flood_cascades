#!/bin/bash
# Every figure, table and number of the paper and the SI from the data in data/ (level A).
# Run from the repository root after `uv sync`:
#     bash reproduce.sh
# Writes output/figures/, output/tables/ and output/numbers.csv; the last step exits with an
# error if a number does not match the paper.

set -euo pipefail
run() {
    local start=$SECONDS
    echo "== $*"
    uv run python "$@"
    echo "   ($((SECONDS - start)) s)"
}

run figure_2.py
run figure_3.py
run figure_4.py
run figure_5.py
run figure_S1.py
run figure_S2.py
run figure_S3.py
run figure_S4.py
run figure_S5.py
run figure_S6.py
run figure_S7.py
run figure_S8.py
run figure_S9.py
run figure_S10.py
run figure_S11.py
run figure_S12.py
run tables.py
run paper_numbers.py
