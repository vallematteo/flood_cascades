#!/bin/bash
#SBATCH --job-name=cascades_rnet
#SBATCH --output=logs/cascades_rnet_%j.out
#SBATCH --error=logs/cascades_rnet_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=336G
#SBATCH --time=05:00:00

# Foreclosure cascades: the full design (65,520 runs) and its reduction, in one job.
# Submit from the repository root:
#     sbatch experiments/cascades_rnet/slurm_run.sh
#
# Needs Julia 1.10.4 (the environment in Project.toml / Manifest.toml, instantiated) and
# Python 3.11 with requirements.txt. Load them as your cluster requires, for example:
#     module load Julia/1.10.4-linux-x86_64 Python/3.11.3-GCCcore-12.3.0
#
# The raw output (about 1.5 GB) goes to $RAW_DIR; the reduced files go to data/cascades_rnet/.
# On a 192-core node (AMD Genoa) the simulation took TODO hours.

set -euo pipefail
RAW_DIR="${RAW_DIR:-output/cascades_rnet/raw}"
mkdir -p logs "$RAW_DIR"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"

# 1. Simulation: 13 cells, one after the other, on all cores of the node
julia --project=. experiments/cascades_rnet/run.jl \
    --num-workers "${SLURM_NTASKS:-4}" \
    --out-dir "$RAW_DIR"

# 2. Reduction to data/cascades_rnet/
python experiments/cascades_rnet/reduce_runs.py              --raw-dir "$RAW_DIR"
python experiments/cascades_rnet/reconstruct_cascades.py     --raw-dir "$RAW_DIR" --workers 64
python experiments/cascades_rnet/cluster_attribution_free.py --raw-dir "$RAW_DIR" --workers 64
python experiments/cascades_rnet/reduce_fcl_cells.py         --raw-dir "$RAW_DIR"

echo "Done: $(date)"
