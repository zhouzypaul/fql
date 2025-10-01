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

for seed in 1 2 3; do
  python main.py \
    --seed $seed \
    --env_name cube-single-play-singletask-v0 \
    --offline_steps 1000000 \
    --online_steps 500000 \
    --save_dir $REPO_ROOT/exp/ \
    --wandb_run_group sampled_adv_softmax_off_2_on \
    --optimal_var sampled_adv_softmax \
    $@ &
done

wait

for seed in 1 2 3; do
  python main.py \
    --seed $seed \
    --env_name scene-play-singletask-v0 \
    --offline_steps 1000000 \
    --online_steps 500000 \
    --save_dir $REPO_ROOT/exp/ \
    --wandb_run_group sampled_adv_softmax_off_2_on \
    --optimal_var sampled_adv_softmax \
    $@ &
done

wait

for seed in 1 2 3; do
  python main.py \
    --seed $seed \
    --env_name cube-single-play-singletask-v0 \
    --offline_steps 1000000 \
    --online_steps 500000 \
    --save_dir $REPO_ROOT/exp/ \
    --wandb_run_group binary_softmax_loss_off_2_on \
    --optimal_var binary_softmax_loss \
    $@ &
done

wait

for seed in 1 2 3; do
  python main.py \
    --seed $seed \
    --env_name scene-play-singletask-v0 \
    --offline_steps 1000000 \
    --online_steps 500000 \
    --save_dir $REPO_ROOT/exp/ \
    --wandb_run_group binary_softmax_loss_off_2_on \
    --optimal_var binary_softmax_loss \
    $@ &
done

wait