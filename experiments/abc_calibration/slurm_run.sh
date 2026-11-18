#!/bin/bash
#SBATCH --job-name=abc_calibration
#SBATCH --output=logs/abc_calibration_%j.out
#SBATCH --error=logs/abc_calibration_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=336G
#SBATCH --time=02:00:00

# Calibration: the 10,010 runs of output/abc_calibration/run_list.csv and their reduction to
# output/abc_calibration/reduced/runs_set0.csv.gz, in one job. Submit from the repository root, after
#     python experiments/abc_calibration/design.py
# with
#     sbatch experiments/abc_calibration/slurm_run.sh
#
# Needs Julia 1.10.4 (Project.toml / Manifest.toml, instantiated) and the Python environment
# (`uv sync` once, README, Installation guide).
# The runs use the runner of the sensitivity analysis (experiments/global_sa/run.jl).
# SLURM writes the job log to logs/ before the script starts: create it once with `mkdir logs`
# in the repository root before submitting.

set -euo pipefail
RAW_DIR="${RAW_DIR:-output/abc_calibration/raw}"
mkdir -p "$RAW_DIR"
source .venv/bin/activate
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"

julia --project=. experiments/global_sa/run.jl \
    --run-list output/abc_calibration/run_list.csv \
    --num-workers "${SLURM_NTASKS:-4}" \
    --raw-dir "$RAW_DIR"

python experiments/global_sa/reduce.py \
    --run-list output/abc_calibration/run_list.csv \
    --raw-dir "$RAW_DIR" --out-dir output/abc_calibration/reduced --runs-only \
    --workers "${SLURM_NTASKS:-4}"

echo "Done: $(date)"
