import os
import platform

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
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.train import train_bc_agent
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'Whether to run in debug mode.')
flags.DEFINE_string('wandb_run_group', 'Debug', 'Run group.')
flags.DEFINE_string('wandb_project', 'fql', 'Wandb project name.')
flags.DEFINE_boolean('wandb_offline', False, 'Whether to run wandb in offline mode.')
flags.DEFINE_boolean('wandb_log_code', False, 'Whether to log code to wandb.')
flags.DEFINE_multi_string('wandb_tags', None, 'Wandb tags.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('bc_eval_interval', 50000, 'BC evaluation interval.')
flags.DEFINE_integer('bc_eval_episodes', 10, 'Number of episodes for BC evaluation.')
flags.DEFINE_integer('bc_steps', 500000, 'Number of BC steps.')
flags.DEFINE_bool('advantage_gradient', False, 'Whether to use advantage gradient.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_string('optimal_var', 'binary', 'Optimal variable.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def main(_):
    # Make environment and datasets first to get config
    config = FLAGS.agent
    
    config['optimal_var'] = FLAGS.optimal_var
    # Create a more descriptive experiment name
    agent_name = config['agent_name']
    env_short = FLAGS.env_name.replace('-singletask', '').replace('-v0', '').replace('-v1', '').replace('-v2', '')
    timestamp = time.strftime("%m%d_%H%M")
    exp_name = f"{FLAGS.wandb_run_group}_{agent_name}_{env_short}_seed{FLAGS.seed:02d}_{timestamp}"
    

    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    if FLAGS.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
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

    # Create and train BC agent for sampled_adv_softmax
    bc_agent = None
    if config['agent_name'] == 'iql_diffusion' and config['optimal_var'] == 'sampled_adv_softmax':
        from agents.bc import BCAgent, get_config as get_bc_config
        bc_config = get_bc_config()
        bc_agent = BCAgent.create(
            FLAGS.seed + 1,  # Different seed for BC agent
            example_batch['observations'],
            example_batch['actions'],
            bc_config,
        )
        
        # Train BC agent separately
        bc_agent = train_bc_agent(bc_agent=bc_agent, 
                                  train_dataset=train_dataset, 
                                  val_dataset=val_dataset,
                                  eval_env=eval_env, 
                                  eval_episodes=FLAGS.bc_eval_episodes,
                                  video_episodes=FLAGS.video_episodes,
                                  video_frame_skip=FLAGS.video_frame_skip,
                                  bc_steps=FLAGS.bc_steps, 
                                  eval_interval=FLAGS.bc_eval_interval, 
                                  wandb_project=FLAGS.wandb_project,
                                  save_dir=FLAGS.save_dir,
                                  run_name=f"BC_policy_{FLAGS.wandb_run_group}_{FLAGS.env_name}_seed{FLAGS.seed:02d}_{timestamp}",
                                  debug=FLAGS.debug,
                                  wandb_offline=FLAGS.wandb_offline,
                                  wandb_log_code=FLAGS.wandb_log_code,
                                  )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        
    # Set up wandb with both flags and agent config
    setup_wandb(
        project=FLAGS.wandb_project, 
        group=FLAGS.wandb_run_group, 
        tags=FLAGS.wandb_tags,
        name=exp_name,
        hyperparam_dict=config.to_dict(),
        mode='disabled' if FLAGS.debug else ('offline' if FLAGS.wandb_offline else 'online'),
        log_code=FLAGS.wandb_log_code,
    )

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'])

            # Add observations_policy for diffusion agents
            if config['agent_name'] == 'iql_diffusion':
                batch['observations_policy'] = batch['observations']

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            elif config['agent_name'] == 'iql_diffusion':
                # Train main agent with BC agent for adv_beta computation
                agent, update_info = agent.update(batch, 
                                                update_critic=i in range(*config['critic_steps']), 
                                                update_actor=i in range(*config['actor_steps']),
                                                bc_agent=bc_agent)
            else:
                agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent.
            if FLAGS.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = replay_buffer.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            elif config['agent_name'] == 'iql_diffusion':
                # Train main agent with BC agent for adv_beta computation
                agent, update_info = agent.update(batch, 
                                                update_critic=i in range(*config['critic_steps']), 
                                                update_actor=i in range(*config['actor_steps']),
                                                bc_agent=bc_agent)
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                # Add observations_policy for diffusion agents
                if config['agent_name'] == 'iql_diffusion':
                    val_batch['observations_policy'] = val_batch['observations']
                    _, val_info = agent.total_loss(val_batch, grad_params=None, bc_agent=bc_agent)
                else:
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0) and i >= config['actor_steps'][0]:
            renders = []
            eval_metrics = {}
            
            # Special evaluation for diffusion agents with multiple cfg values
            if config['agent_name'] == 'iql_diffusion':
                max_return = -np.inf
                
                if FLAGS.advantage_gradient:
                    assert config['optimal_var'] == 'binary', 'Advantage gradient mode is only supported for binary mode.'
                    print('Using advantage gradient mode. Setting cfg weight to 0 and sweeping w_prime values')
                    cfg_values = [0.0, 1.0, 1.5, 3.0, 5.0]
                    optimality_values = [1.0] # Binary
                    w_prime_values = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
                else:
                    # Standard CFG evaluation
                    cfg_values = [1.0, 1.5, 3.0, 5.0]
                    w_prime_values = [0.0]  # No advantage gradient
                    if config['optimal_var'] in ['softmax', 'sampled_adv_softmax']:
                        optimality_values = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
                    else:
                        optimality_values = [1.0]
                
                # Store results for each cfg-optimality-w_prime combination
                cfg_optimality_results = {}
                
                for cfg in cfg_values:
                    cfg_optimality_results[cfg] = {}
                    for o in optimality_values:
                        cfg_optimality_results[cfg][o] = {}
                        for w_prime in w_prime_values:
                            eval_info, trajs, cur_renders = evaluate(
                                agent=agent,
                                env=eval_env,
                                config=config,
                                num_eval_episodes=FLAGS.eval_episodes,
                                num_video_episodes=FLAGS.video_episodes,
                                video_frame_skip=FLAGS.video_frame_skip,
                                cfg=cfg,
                                o=o,
                                w_prime=w_prime,
                            )
                            renders.extend(cur_renders)
                            
                            # Store results for this cfg-optimality-w_prime combination
                            cfg_optimality_results[cfg][o][w_prime] = eval_info
                            
                            # Track max return across all combinations
                            max_return = max(max_return, eval_info['episode.return'])
                    
                    if config['optimal_var'] == 'binary' and not FLAGS.advantage_gradient:
                        # For standard binary CFG mode (not advantage gradient), log single w_prime=0
                        w_prime_used = w_prime_values[0]  # This will be 0.0 in standard mode
                        eval_name = f'evaluation_cfg{cfg}'
                        for k in ['episode.return', 'episode.length', 'success']:
                            eval_metrics[f'{eval_name}/{k}'] = cfg_optimality_results[cfg][1.0][w_prime_used][k]
                        # Log action statistics for binary CFG mode
                        result = cfg_optimality_results[cfg][1.0][w_prime_used]
                        if 'min_action' in result:
                            eval_metrics[f'{eval_name}/min_action'] = result['min_action']
                        if 'max_action' in result:
                            eval_metrics[f'{eval_name}/max_action'] = result['max_action']
                        if 'average_action' in result:
                            eval_metrics[f'{eval_name}/average_action'] = result['average_action']

                # 1. Log overall best performance
                eval_metrics['evaluation/episode.return'] = max_return
                                
                # 2. Find and log the best cfg-optimality-w_prime combination
                best_cfg = None
                best_o = None
                best_w_prime = None
                for cfg in cfg_values:
                    for o in optimality_values:
                        for w_prime in w_prime_values:
                            if cfg_optimality_results[cfg][o][w_prime]['episode.return'] == max_return:
                                best_cfg = cfg
                                best_o = o
                                best_w_prime = w_prime
                                break
                        if best_cfg is not None:
                            break
                    if best_cfg is not None:
                        break
                
                eval_metrics['evaluation/best_cfg'] = best_cfg
                eval_metrics['evaluation/best_optimality'] = best_o
                eval_metrics['evaluation/best_w_prime'] = best_w_prime
                
                # Log action statistics from the best performing configuration
                if best_cfg is not None:
                    best_result = cfg_optimality_results[best_cfg][best_o][best_w_prime]
                    if 'min_action' in best_result:
                        eval_metrics['evaluation/min_action'] = best_result['min_action']
                    if 'max_action' in best_result:
                        eval_metrics['evaluation/max_action'] = best_result['max_action']
                    if 'average_action' in best_result:
                        eval_metrics['evaluation/average_action'] = best_result['average_action']
                
                # 3. Log individual metrics for each cfg-optimality-w_prime combination
                # This allows wandb to aggregate (mean/std) across runs and create custom plots
                if FLAGS.advantage_gradient:
                    # For advantage gradient mode, log w_prime sweep results
                    for cfg in cfg_values:
                        for o in optimality_values:
                            for w_prime in w_prime_values:
                                result = cfg_optimality_results[cfg][o][w_prime]
                                prefix = f"evaluation_cfg{cfg}/o_{o}/w_prime_{w_prime}"
                                
                                # Log each metric separately so wandb can aggregate them
                                eval_metrics[f'{prefix}/episode_return'] = result['episode.return']
                                eval_metrics[f'{prefix}/success_rate'] = result['success']
                                eval_metrics[f'{prefix}/episode_length'] = result['episode.length']
                                # Log action statistics
                                if 'min_action' in result:
                                    eval_metrics[f'{prefix}/min_action'] = result['min_action']
                                if 'max_action' in result:
                                    eval_metrics[f'{prefix}/max_action'] = result['max_action']
                                if 'average_action' in result:
                                    eval_metrics[f'{prefix}/average_action'] = result['average_action']
                elif config['optimal_var'] != 'binary':
                    # Standard CFG mode
                    for cfg in cfg_values:
                        for o in optimality_values:
                            result = cfg_optimality_results[cfg][o][w_prime_values[0]]  # w_prime=0
                            prefix = f"evaluation_cfg{cfg}/o_{o}"
                            
                            # Log each metric separately so wandb can aggregate them
                            eval_metrics[f'{prefix}/episode_return'] = result['episode.return']
                            eval_metrics[f'{prefix}/success_rate'] = result['success']
                            eval_metrics[f'{prefix}/episode_length'] = result['episode.length']
                            # Log action statistics
                            if 'min_action' in result:
                                eval_metrics[f'{prefix}/min_action'] = result['min_action']
                            if 'max_action' in result:
                                eval_metrics[f'{prefix}/max_action'] = result['max_action']
                            if 'average_action' in result:
                                eval_metrics[f'{prefix}/average_action'] = result['average_action']
                
                # 4. Create heatmap data matrices for visualization
                # For advantage gradient mode, create w_prime vs cfg heatmaps
                if FLAGS.advantage_gradient:
                    returns_matrix = np.array([[cfg_optimality_results[cfg][optimality_values[0]][w_prime]['episode.return'] 
                                              for cfg in cfg_values] for w_prime in w_prime_values])
                    success_matrix = np.array([[cfg_optimality_results[cfg][optimality_values[0]][w_prime]['success'] 
                                              for cfg in cfg_values] for w_prime in w_prime_values])
                    length_matrix = np.array([[cfg_optimality_results[cfg][optimality_values[0]][w_prime]['episode.length'] 
                                             for cfg in cfg_values] for w_prime in w_prime_values])
                else:
                    # Standard CFG mode
                    returns_matrix = np.array([[cfg_optimality_results[cfg][o][w_prime_values[0]]['episode.return'] 
                                              for cfg in cfg_values] for o in optimality_values])
                    success_matrix = np.array([[cfg_optimality_results[cfg][o][w_prime_values[0]]['success'] 
                                              for cfg in cfg_values] for o in optimality_values])
                    length_matrix = np.array([[cfg_optimality_results[cfg][o][w_prime_values[0]]['episode.length'] 
                                             for cfg in cfg_values] for o in optimality_values])
                
                # 5. Create interactive heatmaps (optional - mainly for individual run inspection)
                def create_heatmap(data, title, colorscale='Viridis'):
                    if FLAGS.advantage_gradient:
                        # For advantage gradient mode: w_prime vs cfg
                        x_labels = [f'CFG {cfg}' for cfg in cfg_values]
                        y_labels = [f'w_prime={w}' for w in w_prime_values]
                        hover_template = '<b>CFG</b>: %{x}<br><b>w_prime</b>: %{y}<br><b>Value</b>: %{z:.4f}<extra></extra>'
                        x_title = 'CFG Weight'
                        y_title = 'Advantage Gradient Weight (w_prime)'
                    else:
                        # Standard CFG mode: optimality vs cfg
                        x_labels = [f'CFG {cfg}' for cfg in cfg_values]
                        y_labels = [f'O={o}' for o in optimality_values]
                        hover_template = '<b>CFG</b>: %{x}<br><b>Optimality</b>: %{y}<br><b>Value</b>: %{z:.4f}<extra></extra>'
                        x_title = 'CFG Weight'
                        y_title = 'Optimality Variable'
                        
                    return go.Figure(data=go.Heatmap(
                        z=data,
                        x=x_labels,
                        y=y_labels,
                        colorscale=colorscale,
                        text=[[f'{val:.3f}' for val in row] for row in data],
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        hovertemplate=hover_template
                    )).update_layout(
                        title=f'{title} (Step {i})',
                        xaxis_title=x_title,
                        yaxis_title=y_title,
                        width=600, height=400, font=dict(size=12)
                    )
                
                # Log heatmaps for individual run inspection
                eval_metrics['heatmaps/returns'] = wandb.Plotly(create_heatmap(returns_matrix, 'Episode Returns', 'Viridis'))
                eval_metrics['heatmaps/success'] = wandb.Plotly(create_heatmap(success_matrix, 'Success Rate', 'Blues'))
                eval_metrics['heatmaps/length'] = wandb.Plotly(create_heatmap(length_matrix, 'Episode Length', 'Oranges'))
                
            else:
                # Standard evaluation for other agents
                eval_info, trajs, cur_renders = evaluate(
                    agent=agent,
                    env=eval_env,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                )
                renders.extend(cur_renders)
                for k, v in eval_info.items():
                    eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
