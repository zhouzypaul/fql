export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python main.py \
--agent agents/iql_diffusion.py \
--wandb_project cfgrl \
--seed 0 \
--env_name cube-single-play-singletask-v0 \
--eval_interval 250000 \
--offline_steps 500000 \
--eval_episodes 10 \
$@