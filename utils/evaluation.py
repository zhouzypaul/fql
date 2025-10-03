from collections import defaultdict
import numpy as np
from functools import partial

import jax
import numpy as np
from tqdm import trange


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


def evaluate(
    agent,
    env,
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
    w_prime=None,
):
    """Evaluate the agent in the environment.

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

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    if cfg is not None:
        actor_fn = partial(supply_rng(agent.sample_actions), temperature=0.0, cfg=cfg, o=o, w_prime=w_prime)
    else:
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)
    action_stats = defaultdict(list)  # Track action statistics per episode

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        episode_actions = []  # Track actions for this episode
        should_render = i >= num_eval_episodes

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
            
            # Store action for statistics (before environment step)
            episode_actions.append(action.copy())

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
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
            
            # Compute action statistics for this episode
            if episode_actions:
                episode_actions_array = np.array(episode_actions)
                action_stats['min_action'].append(np.min(episode_actions_array))
                action_stats['max_action'].append(np.max(episode_actions_array))
                action_stats['average_action'].append(np.mean(episode_actions_array))
                # Store all actions for potential further analysis
                action_stats['actions'].extend(episode_actions_array.flatten())
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    
    # Add action statistics to the final stats
    if action_stats:
        stats['min_action'] = np.mean(action_stats['min_action']) if action_stats['min_action'] else 0.0
        stats['max_action'] = np.mean(action_stats['max_action']) if action_stats['max_action'] else 0.0
        stats['average_action'] = np.mean(action_stats['average_action']) if action_stats['average_action'] else 0.0
        # Overall statistics across all actions from all episodes
        if action_stats['actions']:
            all_actions = np.array(action_stats['actions'])
            stats['overall_min_action'] = np.min(all_actions)
            stats['overall_max_action'] = np.max(all_actions)
            stats['overall_average_action'] = np.mean(all_actions)

    return stats, trajs, renders