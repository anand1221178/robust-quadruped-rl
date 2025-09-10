#!/usr/bin/env python3
"""
CLEAN Training Script - Forward Locomotion Only
Simplified version focusing on research proposal requirements
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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor

# SR2L import (if needed)
from agents.ppo_sr2l import PPO_SR2L

# Logging
import wandb
from wandb.integration.sb3 import WandbCallback

# Only the wrappers we actually need
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper, CurriculumDRWrapper

# Import RealAnt environments
import realant_sim

# Suppress warnings
warnings.filterwarnings("ignore", message=".*The environment Ant-v4 is out of date.*")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_env(config: dict, normalize: bool = True, norm_reward: bool = True):
    """Create environment - SIMPLIFIED for forward locomotion"""
    env_name = config.get('env', {}).get('name', 'RealAntMujoco-v0')
    
    # Simple options
    use_success_reward = config.get('env', {}).get('use_success_reward', False)
    use_domain_randomization = config.get('env', {}).get('use_domain_randomization', False)
    
    def make_env():
        env = gym.make(env_name)
        
        # Apply reward wrapper
        if use_success_reward:
            print("✅ Success Reward Wrapper: Forward locomotion training")
            env = SuccessRewardWrapper(env)
        
        # Apply domain randomization (choose between basic and curriculum)
        if use_domain_randomization:
            dr_config = config.get('domain_randomization', {})
            
            # 🔥 CHECK WRAPPER TYPE PREFERENCE FROM CONFIG
            wrapper_type = dr_config.get('wrapper_type', 'auto')
            has_curriculum = any(key.startswith('phase_') for key in dr_config.keys())
            use_curriculum = dr_config.get('use_curriculum', True)
            
            # Determine which wrapper to use
            if wrapper_type == 'CurriculumDRWrapper' or (wrapper_type == 'auto' and has_curriculum and use_curriculum):
                print("🔥 ULTIMATE 3-PHASE CURRICULUM DR: Research proposal compliant! 🔥")
                print("📚 Using CurriculumDRWrapper for phase-based training")
                env = CurriculumDRWrapper(env, dr_config)
            elif wrapper_type == 'DomainRandomizationWrapper' or (wrapper_type == 'auto' and not use_curriculum):
                print("🎲 Basic Domain Randomization: Joint failure robustness")
                print(f"  Joint dropout prob: {dr_config.get('joint_dropout_prob', 0.1)}")
                print(f"  Max dropped joints: {dr_config.get('max_dropped_joints', 2)}")
                env = DomainRandomizationWrapper(env, dr_config)
            else:
                # Default behavior - auto-detect based on phases
                if has_curriculum:
                    print("🔥 ULTIMATE 3-PHASE CURRICULUM DR: Research proposal compliant! 🔥")
                    print("📚 Phase-based training detected - using CurriculumDRWrapper")
                    env = CurriculumDRWrapper(env, dr_config)
                else:
                    print("🎲 Basic Domain Randomization: Joint failure robustness")
                    print(f"  Joint dropout prob: {dr_config.get('joint_dropout_prob', 0.1)}")
                    print(f"  Max dropped joints: {dr_config.get('max_dropped_joints', 2)}")
                    env = DomainRandomizationWrapper(env, dr_config)
            
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.)
    
    return env


def train(config: dict):
    """Main training function"""
    
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
    
    # Save config
    with open(f"{save_path}/config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Experiment: {experiment_config.get('name', 'unknown')}")
    print(f"Environment: {config.get('env', {}).get('name', 'RealAntMujoco-v0')}")
    print(f"Total timesteps: {config.get('total_timesteps', 10000000):,}")
    
    # Create environment
    env = create_env(config, normalize=True, norm_reward=True)
    
    # 🔥 CHECK FOR PRETRAINED MODEL LOADING
    pretrained_model_path = config.get('pretrained_model')
    pretrained_vec_normalize_path = config.get('pretrained_vec_normalize')
    
    if pretrained_model_path:
        print(f"🔄 FINE-TUNING MODE: Loading pretrained model from {pretrained_model_path}")
        
        # Load pretrained VecNormalize if specified
        if pretrained_vec_normalize_path:
            print(f"📊 Loading pretrained VecNormalize from {pretrained_vec_normalize_path}")
            env = VecNormalize.load(pretrained_vec_normalize_path, env)
            env.training = True  # Enable training mode for fine-tuning
            env.norm_reward = True
        
        # Load pretrained model
        print(f"🧠 Loading pretrained model weights...")
        model = PPO.load(pretrained_model_path, env=env, tensorboard_log=f"{save_path}/tensorboard/")
        
        # Update learning rate for fine-tuning if specified
        if 'learning_rate' in config.get('ppo', {}):
            model.learning_rate = config['ppo']['learning_rate']
            print(f"📉 Updated learning rate to {model.learning_rate} for fine-tuning")
        
        print("✅ Pretrained model loaded successfully for fine-tuning!")
        
    else:
        # Check if using SR2L
        use_sr2l = config.get('sr2l', {}).get('enabled', False)
        
        if use_sr2l:
            print("🔬 Using SR2L algorithm")
            sr2l_config = config.get('sr2l', {})
            model = PPO_SR2L(
                "MlpPolicy", 
                env,
                sr2l_config=sr2l_config,
                verbose=1,
                tensorboard_log=f"{save_path}/tensorboard/",
                **config.get('ppo', {})
            )
        else:
            print("🏃 Using standard PPO")
            # Get activation function
            activation_name = config.get('policy', {}).get('activation', 'relu')
            activation_fn = nn.ReLU if activation_name == 'relu' else nn.Tanh
            
            # Get network architecture
            net_arch = config.get('policy', {}).get('hidden_sizes', [64, 128])
            
            model = PPO(
                "MlpPolicy", 
                env,
                verbose=1,
                tensorboard_log=f"{save_path}/tensorboard/",
                policy_kwargs=dict(
                    activation_fn=activation_fn,
                    net_arch=[dict(pi=net_arch, vf=net_arch)]
                ),
                **config.get('ppo', {})
            )
    
    # Setup callbacks
    callbacks = []
    
    # Progress bar callback - SHOWS ACTUAL TRAINING PROGRESS
    progress_callback = ProgressBarCallback()
    callbacks.append(progress_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('save_freq', 100000),
        save_path=f"{save_path}/checkpoints/",
        name_prefix="checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # W&B callback
    if config.get('logging', {}).get('wandb', False):
        wandb_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_path=f"{save_path}/models/"
        )
        callbacks.append(wandb_callback)
    
    # Start training
    print(f"\n🚀 Starting training for {config.get('total_timesteps', 10000000):,} timesteps...")
    print("=" * 60)
    
    model.learn(
        total_timesteps=config.get('total_timesteps', 10000000),
        callback=callbacks,
        tb_log_name="ppo"
    )
    
    # Save final model
    print("\n💾 Saving final model...")
    model.save(f"{save_path}/final_model")
    env.save(f"{save_path}/vec_normalize.pkl")
    
    # Save best model if evaluation was done
    if hasattr(model, 'best_model_save_path'):
        print(f"💫 Best model saved at: {model.best_model_save_path}")
    
    print(f"\n✅ Training complete! Results saved to: {save_path}")
    
    if config.get('logging', {}).get('wandb', False):
        wandb.finish()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train robust quadruped models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Start training
    train(config)


if __name__ == "__main__":
    main()