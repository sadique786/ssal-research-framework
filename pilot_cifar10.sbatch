#!/bin/bash
#SBATCH --job-name=ssal-cifar-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/pilot-%j.out
#SBATCH --error=logs/pilot-%j.err

set -euo pipefail
cd "$HOME/ssal-research-framework"
source "$HOME/venv/bin/activate"

echo "Start: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Effective configuration:"
python -m ssal.main --cfg job --resolve experiment.training.epochs=1 active_learning.initial_labeled_size=500 active_learning.query_size=100 active_learning.num_rounds=2

echo "Starting training..."
/usr/bin/time -v python -u -m ssal.main experiment.training.epochs=1 active_learning.initial_labeled_size=500 active_learning.query_size=100 active_learning.num_rounds=2

echo "End: $(date --iso-8601=seconds)"
