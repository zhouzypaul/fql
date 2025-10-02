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


python rollout.py \
    --agent agents/iql_diffusion.py \
    --seed 1 \
    --env_name cube-single-play-singletask-v0 \
    --eval_episodes 10 \
    --eval_batch_size 10 \
    --video_episodes 10 \
    --restore_path $REPO_ROOT/exp/cfgrl/sampled_adv_softmax_o_sequential/sampled_adv_softmax_o_sequential_iql_diffusion_cube-single-play_seed01_0929_1826 \
    --restore_epoch 1000000 \
    --optimal_var sampled_adv_softmax \
    $@