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

python main.py \
  --agent agents/iql_diffusion.py \
  --wandb_project cfgrl \
  --seed 1 \
  --env_name cube-single-play-singletask-v0 \
  --restore_path /home/charles-xu/code/fql/exp/cfgrl/binary_o/binary_o_iql_diffusion_cube-single-play_seed01_0919_1805 \
  --restore_epoch 1000000 \
  --offline_steps 0 \
  --online_steps 500000 \
  --eval_interval 10000 \
  --save_interval 100000 \
  --eval_episodes 10 \
  --eval_batch_size 10 \
  --save_dir $REPO_ROOT/exp/ \
  --wandb_log_code True \
  --wandb_run_group sampled_adv_softmax_online \
  --optimal_var sampled_adv_softmax \
  $@ &

python main.py \
  --agent agents/iql_diffusion.py \
  --wandb_project cfgrl \
  --seed 2 \
  --env_name cube-single-play-singletask-v0 \
  --restore_path /home/charles-xu/code/fql/exp/cfgrl/binary_o/binary_o_iql_diffusion_cube-single-play_seed02_0919_1821 \
  --restore_epoch 1000000 \
  --offline_steps 0 \
  --online_steps 500000 \
  --eval_interval 10000 \
  --save_interval 100000 \
  --eval_episodes 10 \
  --eval_batch_size 10 \
  --save_dir $REPO_ROOT/exp/ \
  --wandb_log_code True \
  --wandb_run_group sampled_adv_softmax_online \
  --optimal_var sampled_adv_softmax \
  $@ &

python main.py \
  --agent agents/iql_diffusion.py \
  --wandb_project cfgrl \
  --seed 3 \
  --env_name cube-single-play-singletask-v0 \
  --restore_path /home/charles-xu/code/fql/exp/cfgrl/binary_o/binary_o_iql_diffusion_cube-single-play_seed03_0919_1837 \
  --restore_epoch 1000000 \
  --offline_steps 0 \
  --online_steps 500000 \
  --eval_interval 10000 \
  --save_interval 100000 \
  --eval_episodes 10 \
  --eval_batch_size 10 \
  --save_dir $REPO_ROOT/exp/ \
  --wandb_log_code True \
  --wandb_run_group sampled_adv_softmax_online \
  --optimal_var sampled_adv_softmax \
  $@ &

wait