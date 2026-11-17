#!/bin/bash
#SBATCH --job-name=global_sa
#SBATCH --output=logs/global_sa_%A_%a.out
#SBATCH --error=logs/global_sa_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=336G
#SBATCH --time=07:00:00
#SBATCH --array=0-4

# Global sensitivity analysis: the 166,400 runs of output/global_sa/run_list.csv in five
# array tasks of 33,280 runs, one node each. Submit from the repository root, after
#     uv run python experiments/global_sa/design.py
# with
#     sbatch experiments/global_sa/slurm_run.sh
# and then the reduction, once all tasks have finished:
#     sbatch --dependency=afterok:<job id> experiments/global_sa/slurm_reduce.sh
#
# Needs Julia 1.10.4 (Project.toml / Manifest.toml, instantiated). Load it as your cluster
# requires, for example:  module load Julia/1.10.4-linux-x86_64
# The raw output (four Parquet files per run, 665,600 files) goes to $RAW_DIR; put it on a
# scratch file system.

set -euo pipefail
RAW_DIR="${RAW_DIR:-output/global_sa/raw}"
mkdir -p logs "$RAW_DIR"
echo "Job ${SLURM_ARRAY_JOB_ID:-local}[${SLURM_ARRAY_TASK_ID:-0}] on $(hostname), $(date)"

julia --project=. experiments/global_sa/run.jl \
    --run-list output/global_sa/run_list.csv \
    --chunk "${SLURM_ARRAY_TASK_ID:-0}" --num-chunks 5 \
    --num-workers "${SLURM_NTASKS:-4}" \
    --raw-dir "$RAW_DIR"

echo "Done: $(date)"
