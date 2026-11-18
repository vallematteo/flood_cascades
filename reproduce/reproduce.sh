#!/bin/bash
# Every figure, table and number of the paper and the SI from the data in data/ (level A).
# Run from the repository root, with the Python environment active (README, Installation guide):
#     bash reproduce/reproduce.sh
# Writes output/figures/, output/tables/ and output/numbers.csv.

set -euo pipefail
cd "$(dirname "$0")/.."
run() {
    local start=$SECONDS
    echo "== $*"
    python "reproduce/$1"
    echo "   ($((SECONDS - start)) s)"
}

run figure_2.py
run figure_3.py
run figure_4.py
run figure_5.py
run figure_SI1.py
run figure_SI2.py
run figure_SI3.py
run figure_SI4.py
run figure_SI5.py
run figure_SI6.py
run figure_SI7.py
run figure_SI8.py
run figure_SI9.py
run figure_SI10.py
run figure_SI11.py
run figure_SI12.py
run tables.py
run paper_numbers.py
