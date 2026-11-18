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
# output/global_sa/reduced/runs_set{0-4}.csv.gz and vectors_set{0-4}.csv. Needs the Python environment
# (`uv sync` once, README, Installation guide).
# SLURM writes the job log to logs/ before the script starts: create it once with `mkdir logs`
# in the repository root before submitting.

set -euo pipefail
RAW_DIR="${RAW_DIR:-output/global_sa/raw}"
source .venv/bin/activate
python experiments/global_sa/reduce.py --raw-dir "$RAW_DIR" --workers "${SLURM_NTASKS:-4}"
echo "Done: $(date)"
