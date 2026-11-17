#!/bin/bash
#SBATCH --job-name=global_sa_reduce
#SBATCH --output=logs/global_sa_reduce_%j.out
#SBATCH --error=logs/global_sa_reduce_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=336G
#SBATCH --time=02:00:00

# Reduction of the global sensitivity analysis: the Parquet output in $RAW_DIR to
# data/global_sa/runs_set{0-4}.csv.gz and vectors_set{0-4}.csv. Needs uv (`uv sync` once).

set -euo pipefail
RAW_DIR="${RAW_DIR:-output/global_sa/raw}"
mkdir -p logs
uv run python experiments/global_sa/reduce.py --raw-dir "$RAW_DIR" --workers "${SLURM_NTASKS:-4}"
echo "Done: $(date)"
