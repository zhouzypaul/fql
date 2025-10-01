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

# sampled_adv_softmax for scene
for critic_steps in 100000 200000 300000 400000; do
    for seed in 1 2 3; do
      total_steps=$((500000 + $critic_steps))
      (
        python main.py \
        --agent agents/iql_diffusion.py \
        --wandb_project cfgrl \
        --seed $seed \
        --env_name scene-play-singletask-v0 \
        --eval_episodes 10 \
        --eval_batch_size 10 \
        --save_dir $REPO_ROOT/exp/ \
        --offline_steps $total_steps \
        --wandb_run_group sampled_adv_softmax_o_sequential_$critic_steps \
        --optimal_var sampled_adv_softmax \
        --critic_pretrain_steps $critic_steps \
        $@
       ) || echo "Run failed"
  done
done

# binary for scene
for critic_steps in 100000 200000 300000 400000; do
    for seed in 1 2 3; do
      total_steps=$((500000 + $critic_steps))
      (
        python main.py \
        --agent agents/iql_diffusion.py \
        --wandb_project cfgrl \
        --seed $seed \
        --env_name scene-play-singletask-v0 \
        --eval_episodes 10 \
        --eval_batch_size 10 \
        --save_dir $REPO_ROOT/exp/ \
        --offline_steps $total_steps \
        --wandb_run_group binary_o_sequential_$critic_steps \
        --optimal_var binary \
        --critic_pretrain_steps $critic_steps \
        $@
       ) || echo "Run failed"
  done
done

# binary_softmax_loss for both envs
for seed in 1 2 3; do
  for critic_steps in 100000 200000 300000 400000; do
      for env in scene-play-singletask-v0 cube-single-play-singletask-v0; do
      total_steps=$((500000 + $critic_steps))
        (
          python main.py \
          --agent agents/iql_diffusion.py \
          --wandb_project cfgrl \
          --seed $seed \
          --env_name $env \
          --eval_episodes 10 \
          --eval_batch_size 10 \
          --save_dir $REPO_ROOT/exp/ \
          --offline_steps $total_steps \
          --wandb_run_group binary_softmax_loss_sequential_$critic_steps \
          --optimal_var binary_softmax_loss \
          --critic_pretrain_steps $critic_steps \
          $@
        ) || echo "Run failed"
    done
  done
done