#!/usr/bin/env python3
"""
Main training script that uses YAML configuration files
This is the proper way to handle configs!
"""

import os
import sys
import gymnasium as gym
import numpy as np
import warnings
from datetime import datetime
import torch.nn as nn
import yaml
from pathlib import Path
import argparse

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# SR2L import
from agents.ppo_sr2l import PPO_SR2L

# Logging
import wandb
from wandb.integration.sb3 import WandbCallback

#Custom success wrapper
from envs.success_reward_wrapper import SuccessRewardWrapper
from utils.custom_callbacks import CustomMetricsCallback

# Import RealAnt environments
import realant_sim

# Suppress deprecation warning
warnings.filterwarnings("ignore", message=".*The environment Ant-v4 is out of date.*")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(*configs):
    """Merge multiple config dictionaries"""
    result = {}
    for config in configs:
        if config:
            result.update(config)
    return result


"""Below used base ants reward - it started sprinting"""
"""def create_env(env_config: dict, normalize: bool = True, norm_reward: bool = True):
    Create environment based on config
    env_name = env_config['env']['name']
    
    def make_env():
        env = gym.make(env_name)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.)
    
    return env"""


def create_env(env_config: dict, normalize: bool = True, norm_reward: bool = True):
    """Create environment based on config"""
    env_name = env_config['env']['name']
    
    # Check if we should use success reward wrapper - FIX THIS LINE
    use_success_reward = env_config['env'].get('use_success_reward', False)
    
    def make_env():
        env = gym.make(env_name)
        
        # Apply success reward wrapper if enabled
        if use_success_reward:
            print("Using Success Reward Wrapper - Training for calm walking!")
            env = SuccessRewardWrapper(env)
            
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.)
    
    return env
def train(config: dict):
    """Main training function using config dict"""
    
    # The config dict already has all the settings merged
    experiment_config = config.get('experiment', {})
    
    # Initialize W&B if enabled
    if config.get('logging', {}).get('wandb', False):
        run = wandb.init(
            project=config.get('logging', {}).get('wandb_project', 'robust-quadruped-rl'),
            entity=config.get('logging', {}).get('wandb_entity'),
            name=experiment_config.get('name', f"ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            config=config,
            tags=experiment_config.get('tags', ['ppo']),
            sync_tensorboard=True,
        )
        run_id = run.id
    else:
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Create save directory
    save_path = f"experiments/{experiment_config.get('name', 'ppo')}_{run_id}"
    os.makedirs(save_path, exist_ok=True)
    
    # Save config for reproducibility
    with open(f"{save_path}/config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Print training configuration header
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    
    # Create environments  
    env_name = config.get('env', {}).get('name', 'RealAntMujoco-v0')
    use_success_reward = config.get('env', {}).get('use_success_reward', False)
    sr2l_config = config.get('sr2l', {})
    
    print(f"Experiment: {experiment_config.get('name', 'unknown')}")
    print(f"Environment: {env_name}")
    print(f"Success Reward Wrapper: {'ENABLED' if use_success_reward else 'DISABLED'}")
    
    if use_success_reward:
        print("\nVelocity Targets:")
        print(f"  - Target velocity: 1.5 m/s (realistic for RealAnt)")
        print(f"  - Minimum velocity: 0.5 m/s") 
        print(f"  - Maximum velocity: 2.5 m/s")
        print(f"  - Stability bonus: ENABLED (penalizes slipping)")
    
    print(f"\nSR2L Configuration:")
    if sr2l_config.get('enabled', False):
        print(f"  - SR2L: ENABLED")
        print(f"  - Lambda: {sr2l_config.get('lambda', 0.01)}")
        print(f"  - Perturbation std: {sr2l_config.get('perturbation_std', 0.05)}")
        print(f"  - Warmup steps: {sr2l_config.get('warmup_steps', 0):,}")
    else:
        print(f"  - SR2L: DISABLED")
    
    print(f"\nTraining:")
    print(f"  - Total timesteps: {config.get('total_timesteps', 1000000):,}")
    print(f"  - Learning rate: {config.get('ppo', {}).get('learning_rate', 0.0003)}")
    print("="*60 + "\n")
    
    # Create env config dict for the create_env function
    env_config_dict = {'env': config.get('env', {'name': env_name})}
    env = create_env(env_config_dict, normalize=True)
    eval_env = create_env(env_config_dict, normalize=True, norm_reward=False)
    
    # Define network architecture
    policy_kwargs = dict(
        net_arch=config.get('policy', {}).get('hidden_sizes', [64, 128]),
        activation_fn=nn.ReLU,
    )
    
    # Check if SR2L is enabled
    sr2l_enabled = config.get('sr2l', {}).get('enabled', False)
    
    # Create PPO model (with or without SR2L)
    if sr2l_enabled:
        print("Creating PPO model with SR2L (Smooth Regularized RL)...")
        model = PPO_SR2L(
            "MlpPolicy",
            env,
            sr2l_config=config.get('sr2l', {}),
            learning_rate=config.get('ppo', {}).get('learning_rate', 3e-4),
            n_steps=config.get('ppo', {}).get('n_steps', 2048),
            batch_size=config.get('ppo', {}).get('batch_size', 2048),
            n_epochs=config.get('ppo', {}).get('n_epochs', 10),
            gamma=config.get('ppo', {}).get('gamma', 0.99),
            gae_lambda=config.get('ppo', {}).get('gae_lambda', 0.95),
            clip_range=config.get('ppo', {}).get('clip_range', 0.2),
            ent_coef=config.get('ppo', {}).get('ent_coef', 0.0),
            vf_coef=config.get('ppo', {}).get('vf_coef', 0.5),
            max_grad_norm=config.get('ppo', {}).get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            verbose=config.get('logging', {}).get('verbose', 1),
            tensorboard_log=f"./tensorboard/{run_id}",
            seed=config.get('seed', 42),
        )
    else:
        print("Creating standard PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get('ppo', {}).get('learning_rate', 3e-4),
            n_steps=config.get('ppo', {}).get('n_steps', 2048),
            batch_size=config.get('ppo', {}).get('batch_size', 2048),
            n_epochs=config.get('ppo', {}).get('n_epochs', 10),
            gamma=config.get('ppo', {}).get('gamma', 0.99),
            gae_lambda=config.get('ppo', {}).get('gae_lambda', 0.95),
            clip_range=config.get('ppo', {}).get('clip_range', 0.2),
            ent_coef=config.get('ppo', {}).get('ent_coef', 0.0),
            vf_coef=config.get('ppo', {}).get('vf_coef', 0.5),
            max_grad_norm=config.get('ppo', {}).get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            verbose=config.get('logging', {}).get('verbose', 1),
            tensorboard_log=f"./tensorboard/{run_id}",
            seed=config.get('seed', 42),
        )
    
    # Print configuration
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    print(f"Environment: {env_name}")
    algorithm_name = "PPO + SR2L" if sr2l_enabled else "PPO"
    print(f"Algorithm: {algorithm_name}")
    if sr2l_enabled:
        print(f"SR2L lambda: {config.get('sr2l', {}).get('lambda', 0.01)}")
        print(f"SR2L perturbation std: {config.get('sr2l', {}).get('perturbation_std', 0.05)}")
    print(f"Total timesteps: {config.get('total_timesteps', 1000000):,}")
    print(f"Learning rate: {config.get('ppo', {}).get('learning_rate', 3e-4)}")
    print(f"Network architecture: {config.get('policy', {}).get('hidden_sizes', [64, 128])}")
    print(f"Save path: {save_path}")
    print("="*60 + "\n")
    
    # Setup callbacks
    callbacks = []
    
    if config.get('logging', {}).get('wandb', False):
        callbacks.append(WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"{save_path}/models",
            verbose=2,
        ))
    
    # Add custom metrics callback if using success reward
    if use_success_reward:
        print("Adding CustomMetricsCallback for success reward tracking")
        callbacks.append(CustomMetricsCallback())
    
    # Checkpoint callback
    callbacks.append(CheckpointCallback(
        save_freq=config.get('save_freq', 50000),
        save_path=f"{save_path}/checkpoints",
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    ))
    
    # Evaluation callback
    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_model",
        log_path=f"{save_path}/eval",
        eval_freq=config.get('eval_freq', 10000),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    ))
    
    # Train
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=config.get('total_timesteps', 1000000),
            callback=callbacks,
            progress_bar=True,
        )
        
        # Save final model
        model.save(f"{save_path}/final_model")
        env.save(f"{save_path}/vec_normalize.pkl")
        
        print(f"\n Training complete! Results saved to {save_path}")
        
    except KeyboardInterrupt:
        print("\n  Training interrupted! Saving current model...")
        model.save(f"{save_path}/interrupted_model")
        env.save(f"{save_path}/vec_normalize.pkl")
    
    finally:
        if config.get('logging', {}).get('wandb', False):
            wandb.finish()
        env.close()
        eval_env.close()


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Train RL agent with config')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment config file')
    parser.add_argument('--override', nargs='*', default=[],
                       help='Override config params (e.g., train.total_timesteps=10000)')
    
    args = parser.parse_args()
    
    # Load base configs
    config = {}
    
    # Load experiment config (which specifies which other configs to use)
    exp_config = load_config(args.config)
    
    # Load referenced configs
    if 'defaults' in exp_config:
        for default_config in exp_config['defaults']:
            # Convert path like "/train/default" to "configs/train/default.yaml"
            config_path = f"configs{default_config}.yaml"
            if os.path.exists(config_path):
                sub_config = load_config(config_path)
                config = merge_configs(config, sub_config)
    
    # Apply experiment-specific config
    config = merge_configs(config, exp_config)
    
    # Apply command-line overrides
    for override in args.override:
        key, value = override.split('=')
        keys = key.split('.')
        
        # Navigate to the right place in config
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value (try to parse as number)
        try:
            current[keys[-1]] = float(value) if '.' in value else int(value)
        except ValueError:
            current[keys[-1]] = value
    
    # Train with the loaded config
    train(config)


if __name__ == "__main__":
    # If run directly without args, show usage
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("  python train.py --config configs/experiments/ppo_baseline.yaml")
        print("  python train.py --config configs/experiments/ppo_baseline.yaml --override train.total_timesteps=10000")
    else:
        main()