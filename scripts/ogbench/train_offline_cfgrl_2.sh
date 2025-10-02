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


# sweep over softmax beta for sampled_adv_softmax
for env in cube-single-play-singletask-v0; do
  for seed in 1 2 3; do
    for softmax_beta in 3.0 5.0; do
        (
          python main.py \
          --agent agents/iql_diffusion.py \
          --wandb_project cfgrl \
          --seed $seed \
          --env_name $env \
          --save_dir $REPO_ROOT/exp/ \
          --wandb_run_group sampled_adv_softmax_o_${softmax_beta}\
          --optimal_var sampled_adv_softmax \
          --softmax_beta $softmax_beta \
          $@
        ) || echo "Run failed"
    done
  done
done

# sweep over softmax beta for sampled_adv_softmax
for seed in 1 2 3; do
  for env in cube-double-play-singletask-v0 scene-play-singletask-v0; do
    for softmax_beta in 1.0 3.0 5.0; do
        (
          python main.py \
          --agent agents/iql_diffusion.py \
          --wandb_project cfgrl \
          --seed $seed \
          --env_name $env \
          --save_dir $REPO_ROOT/exp/ \
          --wandb_run_group sampled_adv_softmax_o_${softmax_beta}\
          --optimal_var sampled_adv_softmax \
          --softmax_beta $softmax_beta \
          $@
        ) || echo "Run failed"
    done
  done
done