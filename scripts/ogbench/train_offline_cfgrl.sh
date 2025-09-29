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

for seed in 1; do
    (
      python main.py \
      --agent agents/iql_diffusion.py \
      --wandb_project cfgrl \
      --seed $seed \
      --env_name cube-single-play-singletask-v0 \
      --eval_episodes 10 \
      --eval_batch_size 10 \
      --save_dir $REPO_ROOT/exp/ \
      --wandb_run_group sampled_adv_softmax_o \
      --optimal_var sampled_adv_softmax \
      $@
    ) || echo "Run failed"
done