import os
import platform
import imageio
import json
import random
import time

import jax
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import tqdm
import wandb
import gymnasium as gym
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import rollout, flatten
from utils.flax_utils import restore_agent

FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'Whether to run in debug mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', None, 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', 0, 'Restore epoch.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')


flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('eval_batch_size', 10, 'Evaluation batch size.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_string('optimal_var', 'binary', 'Optimal variable.')
flags.DEFINE_float('awr_temperature', 1.0, 'AWR temperature.')
flags.DEFINE_float('softmax_beta', 1.0, 'Softmax beta.')

config_flags.DEFINE_config_file('agent', 'agents/iql_diffusion.py', lock_config=False)


def main(_):
    # Make environment and datasets first to get config
    assert FLAGS.restore_path is not None
    config = FLAGS.agent
    
    config['optimal_var'] = FLAGS.optimal_var
    config['awr_temperature'] = FLAGS.awr_temperature
    config['softmax_beta'] = FLAGS.softmax_beta
    # Create a more descriptive experiment name
    agent_name = config['agent_name']
    env_short = FLAGS.env_name.replace('-singletask', '').replace('-v0', '').replace('-v1', '').replace('-v2', '')
    timestamp = time.strftime("%m%d_%H%M")
    if FLAGS.save_dir is not None:
        save_dir = os.path.join(FLAGS.save_dir, "rollouts", os.path.basename(FLAGS.restore_path))
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.join(FLAGS.restore_path, 'rollouts')
        os.makedirs(save_dir, exist_ok=True)
    if FLAGS.video_episodes > 0:
        videos_dir = os.path.join(FLAGS.restore_path, 'videos')
        os.makedirs(videos_dir, exist_ok=True)

    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    if FLAGS.eval_batch_size > 1:
        assert FLAGS.eval_episodes % FLAGS.eval_batch_size == 0, 'eval_episodes must be divisible by eval_batch_size'
        env_fns = [lambda: make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)[1] for _ in range(FLAGS.eval_batch_size)]
        venv = gym.vector.AsyncVectorEnv(env_fns)
    else:
        venv = None
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    # Use the training dataset as the replay buffer.
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
    replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    # Restore agent.
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Evaluate agent.
    eval_metrics = {}
    
    # Special evaluation for diffusion agents with multiple cfg values
    if config['agent_name'] == 'iql_diffusion':
        max_return = -np.inf
        # cfg_values = [1.0, 1.5]
        cfg_values = [1.0, 1.5, 3.0, 5.0, 10.0, 30.0]
        if config['optimal_var'] in ['softmax', 'sampled_adv_softmax']:
            # optimality_values = [0.5, 0.55]
            optimality_values = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
        else:
            optimality_values = [1.0]
        
        # Store results for each cfg-optimality combination
        cfg_optimality_results = {}
        renders = {}
        
        for cfg in tqdm.tqdm(cfg_values, desc='Evaluating various cfg values'):
            cfg_optimality_results[cfg] = {}
            renders[cfg] = {}
            for o in optimality_values:
                eval_info, trajs, cur_renders = rollout(
                    agent=agent,
                    env=eval_env,
                    venv=venv,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    cfg=cfg,
                    o=o,
                )
                renders[cfg][o] = cur_renders
                
                # Store results for this cfg-optimality combination
                cfg_optimality_results[cfg][o] = eval_info
                
                # Track max return across all combinations
                max_return = max(max_return, eval_info['episode.return'])
                                            
        # 1. Log overall best performance
        eval_metrics['evaluation/episode.return'] = max_return
        eval_metrics['evaluation/success_rate'] = max(cfg_optimality_results[cfg][o]['success'] for cfg in cfg_values for o in optimality_values)
        eval_metrics['evaluation/episode_length'] = min(cfg_optimality_results[cfg][o]['episode.length'] for cfg in cfg_values for o in optimality_values)
                        
        # 2. Find and log the best cfg-optimality combination
        best_cfg = None
        best_o = None
        for cfg in cfg_values:
            for o in optimality_values:
                if cfg_optimality_results[cfg][o]['episode.return'] == max_return:
                    best_cfg = cfg
                    best_o = o
                    break
            if best_cfg is not None:
                break
        
        eval_metrics['evaluation/best_cfg'] = best_cfg
        eval_metrics['evaluation/best_optimality'] = best_o
        
        # 3. Log individual metrics for each cfg-optimality combination
        for cfg in cfg_values:
            for o in optimality_values:
                result = cfg_optimality_results[cfg][o]
                prefix = f"evaluation_cfg{cfg}/o_{o}"
                eval_metrics[f'{prefix}/episode_return'] = result['episode.return']
                eval_metrics[f'{prefix}/success_rate'] = result['success']
                eval_metrics[f'{prefix}/episode_length'] = result['episode.length']
        
        # 4. Create heatmap data matrices for visualization
        returns_matrix = np.array([[cfg_optimality_results[cfg][o]['episode.return'] 
                                    for cfg in cfg_values] for o in optimality_values])
        success_matrix = np.array([[cfg_optimality_results[cfg][o]['success'] 
                                    for cfg in cfg_values] for o in optimality_values])
        length_matrix = np.array([[cfg_optimality_results[cfg][o]['episode.length'] 
                                    for cfg in cfg_values] for o in optimality_values])
        
        def create_heatmap(data, title, cmap='viridis'):
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(data, cmap=cmap)
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(cfg_values)))
            ax.set_yticks(np.arange(len(optimality_values)))
            ax.set_xticklabels([f'CFG {cfg}' for cfg in cfg_values])
            ax.set_yticklabels([f'O={o}' for o in optimality_values])
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            for i in range(len(optimality_values)):
                for j in range(len(cfg_values)):
                    text = ax.text(j, i, f'{data[i, j]:.3f}',
                                 ha="center", va="center", color="w")
            
            ax.set_title(f'{title} (Step {FLAGS.restore_epoch})')
            ax.set_xlabel('CFG Weight')
            ax.set_ylabel('Optimality Variable')
            
            plt.tight_layout()
            return fig
            
        # Create and save heatmaps
        returns_fig = create_heatmap(returns_matrix, 'Episode Returns')
        returns_fig.savefig(os.path.join(save_dir, f'returns_heatmap_{timestamp}.png'))
        plt.close(returns_fig)
        
        success_fig = create_heatmap(success_matrix, 'Success Rate', cmap='Blues') 
        success_fig.savefig(os.path.join(save_dir, f'success_heatmap_{timestamp}.png'))
        plt.close(success_fig)
        
        length_fig = create_heatmap(length_matrix, 'Episode Length', cmap='Oranges')
        length_fig.savefig(os.path.join(save_dir, f'length_heatmap_{timestamp}.png'))
        plt.close(length_fig)
        
        # 5. Save rollout videos
        for cfg in cfg_values:
            for o in optimality_values:
                for idx in range(len(renders[cfg][o]) // 10):
                    max_length = max(len(render) for render in renders[cfg][o][idx*10:(idx+1)*10])
                    l, h, w, c = renders[cfg][o][0].shape
                    video = np.zeros((max_length, h, w*10, c), dtype=np.uint8)
                    for i in range(10):
                        length = len(renders[cfg][o][idx*10+i])
                        video[:length, :, i*w:(i+1)*w, :] = renders[cfg][o][idx*10+i]
                    video_path = os.path.join(videos_dir, f'video_cfg_{cfg}_o_{o}_idx_{idx}_{timestamp}.mp4')
                    imageio.mimsave(video_path, video, fps=30, quality=8, macro_block_size=1)
                    
        print(eval_metrics)
    else:
        # Standard evaluation for other agents
        eval_info, trajs, renders = rollout(
            agent=agent,
            env=eval_env,
            venv=venv,
            config=config,
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
        )
        
        for idx in range(len(renders // 10)):
            video_path = os.path.join(videos_dir, f'video_{idx}_{timestamp}.mp4')
            imageio.mimsave(video_path, np.concatenate(renders[idx*10:(idx+1)*10], axis=1), fps=30, quality=8, macro_block_size=1)

    if save_dir is not None:
        for traj in trajs:
            if "episode" in traj['info'][-1]:
                traj['info'][-1].pop("episode")
        np.save(os.path.join(save_dir, f'{FLAGS.eval_episodes}_episodes_{timestamp}.npy'), trajs)

if __name__ == '__main__':
    app.run(main)