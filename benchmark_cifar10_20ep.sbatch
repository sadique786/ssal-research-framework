#!/bin/bash
#SBATCH --job-name=ssal-cifar-20ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/benchmark20-%j.out
#SBATCH --error=logs/benchmark20-%j.err

set -euo pipefail
cd "$HOME/ssal-research-framework"
source "$HOME/venv/bin/activate"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="l0_cifar10_bench20_${RUN_TAG}"

echo "Start: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Experiment name: ${EXPERIMENT_NAME}"

echo "===== EFFECTIVE CONFIGURATION ====="
python -m ssal.main --cfg job --resolve \
  "experiment.name=${EXPERIMENT_NAME}" \
  experiment.training.epochs=20 \
  active_learning.initial_labeled_size=500 \
  active_learning.query_size=100 \
  active_learning.num_rounds=2

echo "===== TRAINING ====="
/usr/bin/time -f 'TRAIN_WALL_SECONDS=%e\nTRAIN_CPU_SECONDS=%U' \
  python -u -m ssal.main \
  "experiment.name=${EXPERIMENT_NAME}" \
  experiment.training.epochs=20 \
  active_learning.initial_labeled_size=500 \
  active_learning.query_size=100 \
  active_learning.num_rounds=2

echo "End: $(date --iso-8601=seconds)"
