#!/bin/bash
# Demo: the 20 runs of the calibrated parametrisation without floods (S0, delta = 0) and their
# reduction. They give the annual foreclosure rate (0.42%) and the share of households in
# financial distress (0.51%) of the main text, and reproduce data/abc_calibration/calibrated/.
#
# Run from the repository root:  bash demo.sh [workers]   (default 4 workers)
# The runs must use Julia 1.10.4; with juliaup and another default version:
#   JULIA="julia +1.10.4" bash demo.sh
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${1:-4}"
JULIA="${JULIA:-julia}"
OUT=output/demo

JV=$($JULIA -e 'print(VERSION)')
[ "$JV" = "1.10.4" ] || echo "WARNING: Julia $JV; the reference output was produced with Julia 1.10.4, other versions give slightly different runs"

uv run python experiments/abc_calibration/design.py
$JULIA --project=. experiments/global_sa/run.jl \
    --run-list output/abc_calibration/run_list_calibrated.csv --raw-dir "$OUT/raw" --num-workers "$WORKERS"
uv run python experiments/global_sa/reduce.py \
    --run-list output/abc_calibration/run_list_calibrated.csv --raw-dir "$OUT/raw" \
    --out-dir "$OUT/reduced" --runs-only --workers "$WORKERS"

uv run python - <<'EOF'
import pandas as pd
new = pd.read_csv("output/demo/reduced/runs_set0.csv.gz")
ref = pd.read_csv("data/abc_calibration/calibrated/runs_set0.csv.gz")
print(f"\n{len(new)} runs: annual foreclosure rate {new.rate_tot.mean():.2f}% "
      f"(paper 0.42%), households in financial distress {new.arrears_fin.mean():.2f}% (paper 0.51%)")
cols = ["run_id", "n_fcl_fin", "n_fcl_strat", "n_hhyr_fin"]
same = new[cols].equals(ref[cols]) and ((new.rate_tot - ref.rate_tot).abs().max() < 1e-12)
print("identical to data/abc_calibration/calibrated/runs_set0.csv.gz" if same else
      "differs from data/abc_calibration/calibrated/runs_set0.csv.gz")
EOF
