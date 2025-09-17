export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python iql_diffusion.py \
--wandb.project cfgrl \
--seed 0 \
--env_name scene-play-singletask-v0 \
--eval_interval 250000 \
$@