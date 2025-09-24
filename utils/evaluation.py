from collections import defaultdict
import copy
import numpy as np
from functools import partial
from tqdm import trange
import jax


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


class SingleEnvBatchAdapter:
    """Adapter to present a single Gymnasium env as a batched (num_envs=1) env.

    Accepts batched actions with leading batch dimension 1 and always returns
    batched outputs, emulating vector env API sufficiently for evaluation.
    """

    def __init__(self, env):
        self._env = env
        self.num_envs = 1

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return np.expand_dims(obs, axis=0), info

    def step(self, actions):
        action = actions[0] if isinstance(actions, np.ndarray) and actions.ndim >= 1 and actions.shape[0] == 1 else actions
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            np.expand_dims(obs, axis=0),
            np.array([reward], dtype=np.float32),
            np.array([terminated], dtype=np.bool_),
            np.array([truncated], dtype=np.bool_),
            info,
        )

    def render(self):
        return self._env.render()


def _ensure_batch(x):
    if isinstance(x, tuple):
        raise TypeError("Expected observation to be array-like, got tuple.")
    if isinstance(x, np.ndarray) and x.ndim > 0:
        return x
    return np.expand_dims(x, axis=0)


def _vector_infos_to_list(infos, num_envs):
    if isinstance(infos, dict) and any(k.startswith('_') for k in infos):
        per_env = [dict() for _ in range(num_envs)]
        for key, value in infos.items():
            if key.startswith('_'):
                continue
            mask = infos.get(f"_{key}")
            if mask is None:
                mask = np.ones(num_envs, dtype=bool)
            for idx in range(num_envs):
                if mask[idx]:
                    per_env[idx][key] = value[idx]
        return per_env
    if isinstance(infos, dict):
        return [infos for _ in range(num_envs)]
    if isinstance(infos, (list, tuple)):
        return list(infos)
    if infos is None:
        return [dict() for _ in range(num_envs)]
    return [{"info": infos} for _ in range(num_envs)]


def _prepare_actor(agent, cfg, o):
    if cfg is not None:
        return partial(supply_rng(agent.sample_actions), temperature=0.0, cfg=cfg, o=o)
    return supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))


def run_episodes(
    agent,
    env,
    task_id,
    config,
    eval_temperature,
    eval_gaussian,
    cfg,
    o,
    goal_conditioned,
    should_render=False,
    video_frame_skip=3,
):
    """Shared rollout for sequential and vectorized environments (batch-first).

    Always treat inputs/outputs as batched. If `env` is single, use SingleEnvBatchAdapter.
    """

    if not hasattr(env, 'num_envs'):
        env = SingleEnvBatchAdapter(env)
    if env.num_envs > 1 and should_render:
        raise ValueError('Rendering is only supported for single environments.')

    actor_fn = _prepare_actor(agent, cfg, o)

    observations, reset_infos = env.reset(options=dict(task_id=task_id, render_goal=should_render))

    num_envs = env.num_envs
    policy_observations = copy.deepcopy(observations)

    # Per-env state
    active = np.ones(num_envs, dtype=bool)
    returns = np.zeros(num_envs, dtype=np.float32)
    lengths = np.zeros(num_envs, dtype=np.int32)
    trajectories = [defaultdict(list) for _ in range(num_envs)]
    renders = [[] for _ in range(num_envs)]

    # Goals
    infos_per_env = _vector_infos_to_list(reset_infos, num_envs) if isinstance(reset_infos, dict) else ([reset_infos] * num_envs)
    goals = [info.get('goal') for info in infos_per_env]
    goal_frames = [info.get('goal_rendered') for info in infos_per_env]

    rng = np.random.default_rng()

    while not np.all(~active):
        actor_kwargs = dict(observations=policy_observations, temperature=eval_temperature)
        if goal_conditioned:
            actions = actor_fn(observations=policy_observations, temperature=eval_temperature, goals=goals)
        else:
            actions = actor_fn(observations=policy_observations, temperature=eval_temperature)
        actions = np.array(actions)
        if actions.ndim == 0:
            actions = np.array([actions])
        elif actions.ndim == 1:
            actions = actions[None, ...]
        if eval_gaussian is not None:
            actions = rng.normal(loc=actions, scale=eval_gaussian)
        actions = np.clip(actions, -1, 1)

        next_observations, rewards, terminations, truncations, step_infos = env.step(actions)
        infos_per_step = _vector_infos_to_list(step_infos, num_envs)
        done_now = np.logical_or(terminations, truncations)

        for idx in range(num_envs):
            reward = rewards[idx]
            info = infos_per_step[idx]
            next_observation = next_observations[idx]

            if active[idx]:
                lengths[idx] += 1
                returns[idx] += reward

                if done_now[idx] and 'final_observation' in info:
                    next_observation = info['final_observation']
                if 'goal' in info:
                    goals[idx] = info['goal']
                if should_render and 'goal_rendered' in info:
                    goal_frames[idx] = info['goal_rendered']

                transition = dict(
                    observation=policy_observations[idx],
                    next_observation=next_observation,
                    action=actions[idx],
                    reward=reward,
                    done=done_now[idx],
                    info=info,
                )
                add_to(trajectories[idx], transition)

                if should_render:
                    step_count = lengths[idx]
                    if step_count % video_frame_skip == 0 or done_now[idx]:
                        frame = env.render().copy()
                        goal_frame = goal_frames[idx]
                        if goal_frame is not None:
                            renders[idx].append(np.concatenate([goal_frame, frame], axis=0))
                        else:
                            renders[idx].append(frame)

                if done_now[idx]:
                    active[idx] = False

            policy_observations[idx] = next_observation

    # Build final infos per env by taking the last step info in the trajectory if present
    final_infos = []
    for idx in range(num_envs):
        last_info = trajectories[idx]['info'][-1]
        final_info = last_info.get('final_info', last_info)
        final_infos.append(flatten(final_info) if isinstance(final_info, dict) else {})

    if should_render:
        renders = [np.array(r) for r in renders]

    return trajectories, final_infos, renders, returns, lengths


def rollout(
    agent,
    env,
    venv=None,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    cfg=None,
    o=None,
    goal_conditioned=False,
):
    """Evaluate the agent in the environment with optimized execution.

    Returns a tuple: (stats, trajectories, renders)
    """
    trajs = []
    stats = defaultdict(list)
    renders = []

    if venv is not None:
        batch_size = venv.num_envs
        assert num_eval_episodes % batch_size == 0
        total_batches = num_eval_episodes // batch_size

        for _ in trange(total_batches):
            traj_batch, infos_batch, _, returns, lengths = run_episodes(
                agent,
                venv,
                task_id,
                config,
                eval_temperature,
                eval_gaussian,
                cfg,
                o,
                goal_conditioned,
                should_render=False,
                video_frame_skip=video_frame_skip,
            )
            for idx in range(batch_size):
                info_flat = {
                    'episode.return': returns[idx],
                    'episode.length': lengths[idx],
                    **flatten(infos_batch[idx]),
                }
                add_to(stats, info_flat)
            trajs.extend(traj_batch)
    else:
        for _ in range(num_eval_episodes):
            traj_batch, infos_batch, _, returns, lengths = run_episodes(
                agent,
                env,
                task_id,
                config,
                eval_temperature,
                eval_gaussian,
                cfg,
                o,
                goal_conditioned,
                should_render=False,
                video_frame_skip=video_frame_skip,
            )
            trajs.append(traj_batch[0])
            info_flat = flatten(infos_batch[0]) if infos_batch[0] else {}
            info_flat = {
                'episode.return': returns[0],
                'episode.length': lengths[0],
                **info_flat,
            }
            add_to(stats, info_flat)

    for _ in range(num_video_episodes):
        _, _, render_batch, _, _ = run_episodes(
            agent,
            env,
            task_id,
            config,
            eval_temperature,
            eval_gaussian,
            cfg,
            o,
            goal_conditioned,
            should_render=True,
            video_frame_skip=video_frame_skip,
        )
        renders.append(render_batch[0])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders