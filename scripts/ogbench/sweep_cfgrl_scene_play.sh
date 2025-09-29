#!/usr/bin/env bash
set -uo pipefail
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


for seed in 1 2 3; do
    (
      python main.py \
      --agent agents/iql_diffusion.py \
      --wandb_project cfgrl \
      --seed $seed \
      --env_name cube-single-play-singletask-v0 \
      --eval_interval 100000 \
      --offline_steps 1000000 \
      --save_interval 1000000 \
      --eval_episodes 10 \
      --eval_batch_size 10 \
      --save_dir $REPO_ROOT/exp/ \
      --wandb_log_code True \
      --wandb_run_group binary_softmax_loss_3.0 \
      --optimal_var binary_softmax_loss \
      --softmax_beta 3.0 \
      $@
    ) || echo "Run failed for env: cube-single-play-singletask-v0, softmax_beta: 3.0, seed: $seed"
done

(
  python main.py \
  --agent agents/iql_diffusion.py \
  --wandb_project cfgrl \
  --seed 3 \
  --env_name scene-play-singletask-v0 \
  --eval_interval 100000 \
  --offline_steps 1000000 \
  --save_interval 1000000 \
  --eval_episodes 10 \
  --eval_batch_size 10 \
  --save_dir $REPO_ROOT/exp/ \
  --wandb_log_code True \
  --wandb_run_group binary_softmax_loss_1.0 \
  --optimal_var binary_softmax_loss \
  --softmax_beta 1.0 \
  $@
) || echo "Run failed for env: cube-single-play-singletask-v0, softmax_beta: 3.0, seed: 3"

# Binary softmax loss
for softmax_beta in 2.0 3.0; do
  for seed in 1 2 3; do
      (
        python main.py \
        --agent agents/iql_diffusion.py \
        --wandb_project cfgrl \
        --seed $seed \
        --env_name scene-play-singletask-v0 \
        --eval_interval 100000 \
        --offline_steps 1000000 \
        --save_interval 1000000 \
        --eval_episodes 10 \
        --eval_batch_size 10 \
        --save_dir $REPO_ROOT/exp/ \
        --wandb_log_code True \
        --wandb_run_group binary_softmax_loss_$softmax_beta \
        --optimal_var binary_softmax_loss \
        --softmax_beta $softmax_beta \
        $@
      ) || echo "Run failed for env: scene-play-singletask-v0, softmax_beta: $softmax_beta, seed: $seed"
  done
done
