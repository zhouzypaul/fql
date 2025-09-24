import os
import json
import tqdm
import gymnasium as gym
import wandb

from utils.log_utils import CsvLogger, get_wandb_video, setup_wandb
from utils.evaluation import rollout
from utils.flax_utils import save_agent
from agents.bc import BCAgent, get_config as get_bc_config
from utils.datasets import Dataset

def train_bc_agent(bc_agent: BCAgent, 
                   train_dataset: Dataset, 
                   val_dataset: Dataset, 
                   eval_env: gym.Env, 
                   venv: gym.vector.AsyncVectorEnv,
                   bc_steps: int = 500_000, 
                   log_interval: int = 5000, 
                   eval_interval: int = 50000, 
                   run_name: str = None, 
                   eval_episodes: int = 10, 
                   video_episodes: int = 0, 
                   video_frame_skip: int = 3,
                   wandb_project: str = None,
                   debug: bool = False,
                   wandb_offline: bool = False,
                   wandb_log_code: bool = False,
                   save_dir: str = None,
                   eval_batch_size: int = 10):
    """Train BC agent separately with its own wandb run."""
    
    # Set up separate wandb run for BC training
    bc_config = get_bc_config()
    
    setup_wandb(
        project=wandb_project,
        group="BC_policy",  # Separate group for BC training
        tags=["bc"],
        name=run_name,
        hyperparam_dict=bc_config.to_dict(),
        mode='disabled' if debug else ('offline' if wandb_offline else 'online'),
        log_code=wandb_log_code,
    )
    
    # Create separate save directory for BC agent
    bc_save_dir = os.path.join(save_dir, wandb_project, "bc_agent", run_name)
    os.makedirs(bc_save_dir, exist_ok=True)
    
    # Save BC config
    with open(os.path.join(bc_save_dir, 'bc_config.json'), 'w') as f:
        json.dump(bc_config.to_dict(), f)
    
    # Train BC agent
    bc_logger = CsvLogger(os.path.join(bc_save_dir, 'bc_train.csv'))
    bc_eval_logger = CsvLogger(os.path.join(bc_save_dir, 'bc_eval.csv'))
    
    print(f"Training BC agent for {bc_steps} steps...")
    for i in tqdm.tqdm(range(1, bc_steps + 1), desc="BC Training", smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(bc_config['batch_size'])
        bc_agent, bc_info = bc_agent.update(batch)
        
        # Log training metrics
        if i % log_interval == 0:
            bc_metrics = {f'bc_training/{k}': v for k, v in bc_info.items()}
            bc_metrics['bc_training/step'] = i
            wandb.log(bc_metrics, step=i)
            bc_logger.log(bc_metrics, step=i)
        
        # Evaluate BC agent
        if i % eval_interval == 0:
            eval_metrics = {}
            
            # Validation loss evaluation
            if val_dataset is not None:
                val_batch = val_dataset.sample(bc_config['batch_size'])
                bc_val_loss, bc_val_info = bc_agent.policy_loss(val_batch)
                
                # Log validation metrics
                val_metrics = {f'bc_eval/{k}': v for k, v in bc_val_info.items()}
                eval_metrics.update(val_metrics)
                print(f"BC Val Loss at step {i}: {bc_val_loss:.4f}")
            
            renders = []
            # Standard evaluation for other agents
            eval_info, trajs, cur_renders = rollout(
                agent=bc_agent,
                env=eval_env,
                venv=venv,
                num_eval_episodes=eval_episodes,
                num_video_episodes=video_episodes,
                video_frame_skip=video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            bc_eval_logger.log(eval_metrics, step=i)

    save_agent(bc_agent, bc_save_dir, i)

    bc_logger.close()
    bc_eval_logger.close()
    wandb.finish()
    
    print(f"BC training completed. Agent saved to {bc_save_dir}")
    return bc_agent
