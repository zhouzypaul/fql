#!/usr/bin/env bash
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

# ---- Make one snapshot of the CURRENT code and run everything from it ----
REPO_ROOT="$(pwd)"
TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

# Copy working tree minus junk/cache
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.mypy_cache' \
  --exclude '.pytest_cache' \
  --exclude '.DS_Store' \
  --exclude '*.pyc' \
  --exclude 'exp/' \
  "$REPO_ROOT"/ "$TMP"/

cd "$TMP"

ENV_NAME=${1:-cube-single-play-singletask-v0}
shift || true  # Remove first argument, keep rest

echo "Training advantage gradient agent on environment: $ENV_NAME"
echo "Will sweep w_prime values: [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0] during evaluation"
echo "Will sweep w values: [0.0, 1.0, 1.5, 3.0, 5.0] during evaluation"

for seed in 1 2 3 4 5 6; do
    echo "Starting seed $seed..."
    python main.py \
    --agent agents/iql_diffusion.py \
    --wandb_project advantage_gradient_tuning \
    --seed $seed \
    --env_name $ENV_NAME \
    --eval_interval 500000 \
    --offline_steps 1000000 \
    --save_interval 1000000 \
    --eval_episodes 50 \
    --save_dir $REPO_ROOT/exp/ \
    --wandb_log_code True \
    --wandb_run_group advantage_gradient_wprime \
    --optimal_var binary \
    --advantage_gradient True \
    "$@"
done

echo "All seeds completed!"
echo "Check wandb project 'advantage_gradient_tuning' for results"
echo "Look for heatmaps showing w_prime vs performance"
