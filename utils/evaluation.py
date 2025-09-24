from collections import defaultdict
import numpy as np
import copy
from functools import partial
from tqdm import trange
import time
import gymnasium as gym
import jax
import jax.numpy as jnp


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


def run_single_episode_sequential(agent, env, task_id, config, num_video_episodes, 
                                 video_frame_skip, eval_temperature, eval_gaussian, cfg, o, 
                                 goal_conditioned, should_render):
    """Run a single episode sequentially (non-parallel version).
    
    Returns:
        Tuple of (traj, stats, render)
    """
    # Create actor function for this worker
    if cfg is not None:
        actor_fn = partial(supply_rng(agent.sample_actions), temperature=0.0, cfg=cfg, o=o)
    else:
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    
    traj = defaultdict(list)

    observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
    goal = info.get('goal')
    goal_frame = info.get('goal_rendered')
    done = False
    step = 0
    render = []
    
    while not done:
        if goal_conditioned:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
        else:
            action = actor_fn(observations=observation, temperature=eval_temperature)
        action = np.array(action)
        if eval_gaussian is not None:
            action = np.random.normal(action, eval_gaussian)
        action = np.clip(action, -1, 1)

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if should_render and (step % video_frame_skip == 0 or done):
            frame = env.render().copy()
            if goal_frame is not None:
                render.append(np.concatenate([goal_frame, frame], axis=0))
            else:
                render.append(frame)

        transition = dict(
            observation=observation,
            next_observation=next_observation,
            action=action,
            reward=reward,
            done=done,
            info=info,
        )
        add_to(traj, transition)
        observation = next_observation
    
    return traj, info, render


def run_jax_vectorized_episodes(agent, venv, task_id, config, batch_size, 
                               eval_temperature, eval_gaussian, cfg, o, goal_conditioned):
    """Run episodes using JAX vectorization for maximum speed.
    
    This function uses vectorized action sampling when available for significant speedup.
    
    Returns:
        List of episode results
    """
    observations, infos = venv.reset(options=dict(task_id=task_id, render_goal=False))

    # Track per-env episode state
    steps = np.zeros((batch_size,), dtype=np.int32)
    returns = np.zeros((batch_size,), dtype=np.float32)
    dones = np.zeros((batch_size,), dtype=bool)
    final_infos = [dict() for _ in range(batch_size)]

    # Prepare actor function (batched by default)
    if cfg is not None:
        actor_fn = partial(supply_rng(agent.sample_actions), temperature=0.0, cfg=cfg, o=o)
    else:
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    # Optional goals (may not be provided by vector reset)
    goals = None
    if goal_conditioned:
        assert NotImplementedError("Goal-conditioned evaluation is not implemented for JAX vectorized episodes")

    # Rollout until all envs are done
    zero_action = np.zeros(venv.single_action_space.shape, dtype=venv.single_action_space.dtype)
    while not np.all(dones):
        # Compute batched actions once
        if goal_conditioned and goals is not None:
            actions = actor_fn(observations=observations, goals=goals, temperature=eval_temperature)
        else:
            actions = actor_fn(observations=observations, temperature=eval_temperature)
        actions = np.array(actions)
        if eval_gaussian is not None:
            actions = np.random.normal(actions, eval_gaussian)
        actions = np.clip(actions, -1, 1)

        # For completed envs, send zero actions (ignored)
        if np.any(dones):
            actions = actions.copy()
            actions[dones] = zero_action

        # Step environments
        observations, rewards, terminations, truncations, infos = venv.step(actions)
        done_now = np.logical_or(terminations, truncations)
        
        # Accumulate
        returns += rewards * (~dones)
        steps += (~dones)
        for idx in range(batch_size):
            if done_now[idx] and not dones[idx]:
                final_infos[idx] = flatten(infos['final_info'][idx]) # TODO(Charles): Not sure why there's a final_info field in the infos dictionary
        dones = np.logical_or(dones, done_now)

    # Build results
    batch_results = []
    for idx in range(batch_size):
        result = {
            'episode.return': returns[idx],
            'episode.length': steps[idx],
            'success': final_infos[idx].get('success')
        }
        batch_results.append(result)

    return batch_results


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

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.
        cfg: CFG parameter for diffusion agents.
        o: Optimality parameter for diffusion agents.
        goal_conditioned: Whether the agent is goal-conditioned.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    # Run episodes with optimized batching
    trajs = []
    stats = defaultdict(list)
    renders = []
    
    if venv is not None:
        batch_size = venv.num_envs
        assert num_eval_episodes % batch_size == 0
        
        # Use progress bar for batch processing
        total_batches = num_eval_episodes // batch_size
        
        for batch_num in trange(total_batches):
            batch_infos = run_jax_vectorized_episodes(
                agent, venv, task_id, config, batch_size, 
                eval_temperature, eval_gaussian, cfg, o, goal_conditioned
            )
            # Collect results from batch
            for info in batch_infos:
                add_to(stats, info)
            
    # If venv is not provided, run episodes sequentially
    else:
        for i in range(num_eval_episodes):
            traj, info, render = run_single_episode_sequential(
                agent, env, task_id, config, num_video_episodes, 
                video_frame_skip, eval_temperature, eval_gaussian, cfg, o, 
                goal_conditioned, should_render=False
            )
            add_to(stats, flatten(info))
            trajs.append(traj)
    
    # Handle video episodes separately (these need rendering)
    for i in range(num_video_episodes):
        should_render = True
        traj, info, render = run_single_episode_sequential(
            agent, env, task_id, config, num_video_episodes, 
            video_frame_skip, eval_temperature, eval_gaussian, cfg, o, 
            goal_conditioned, should_render
        )
        renders.append(np.array(render))

    # Compute mean statistics
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders