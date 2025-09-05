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
from envs.target_walking_wrapper import TargetWalkingWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper, CurriculumDRWrapper
from envs.robust_dr_wrapper import RobustDRWrapper
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


def create_env(env_config: dict, normalize: bool = True, norm_reward: bool = True, full_config: dict = None):
    """Create environment based on config"""
    env_name = env_config['env']['name']
    
    # Check wrapper options
    use_success_reward = env_config['env'].get('use_success_reward', False)
    use_target_walking = env_config['env'].get('use_target_walking', False)
    use_domain_randomization = env_config['env'].get('use_domain_randomization', False)
    use_straight_line = env_config['env'].get('use_straight_line', False)
    
    # Check for permanent DR in full config
    use_permanent_dr = False
    permanent_dr_config = {}
    if full_config:
        permanent_dr_config = full_config.get('permanent_dr', {})
        use_permanent_dr = permanent_dr_config.get('enabled', False)
        
    # Check for straight-line config
    straight_line_config = {}
    if full_config:
        straight_line_config = full_config.get('straight_line', {})
    
    def make_env():
        env = gym.make(env_name)
        
        # Apply reward wrapper (mutually exclusive)
        if use_target_walking:
            target_distance = env_config['env'].get('target_distance', 5.0)
            print(f"Using Target Walking Wrapper - Goal-directed navigation ({target_distance}m targets)!")
            env = TargetWalkingWrapper(env, target_distance=target_distance)
        elif use_success_reward:
            print("Using Success Reward Wrapper - Training for fast walking!")
            env = SuccessRewardWrapper(env)
        
        # Apply Permanent DR wrapper if specified (takes precedence)
        if use_permanent_dr:
            print("Using PERMANENT Domain Randomization - Adaptive locomotion with disabilities!")
            print(f"  Max failed joints: {permanent_dr_config.get('max_failed_joints', 4)}")
            print(f"  Failure rate: {permanent_dr_config.get('failure_rate', 0.001)}")
            print(f"  Warmup steps: {permanent_dr_config.get('warmup_steps', 1000000):,}")
            print(f"  Curriculum steps: {permanent_dr_config.get('curriculum_steps', 10000000):,}")
            from envs.permanent_dr_wrapper import PermanentDRCurriculumWrapper
            env = PermanentDRCurriculumWrapper(
                env,
                failure_rate=permanent_dr_config.get('failure_rate', 0.001),
                max_failed_joints=permanent_dr_config.get('max_failed_joints', 4),
                warmup_steps=permanent_dr_config.get('warmup_steps', 1000000),
                curriculum_steps=permanent_dr_config.get('curriculum_steps', 10000000),
                start_failures=permanent_dr_config.get('start_failures', 0),
                end_failures=permanent_dr_config.get('end_failures', 4),
                verbose=permanent_dr_config.get('verbose', False)
            )
        # Apply Domain Randomization wrapper if specified  
        elif use_domain_randomization:
            dr_config = env_config.get('domain_randomization', {})
            wrapper_type = dr_config.get('wrapper_type', 'standard')
            
            if wrapper_type == 'robust':
                print("Using ROBUST Domain Randomization - Advanced fault modeling!")
                print(f"  Fault types: {dr_config.get('fault_types', ['lock', 'weak', 'delay'])}")
                print(f"  Surprise mode: {dr_config.get('surprise_mode', False)}")
                print(f"  Curriculum: {dr_config.get('use_curriculum', False)}")
                env = RobustDRWrapper(env, dr_config)
            elif dr_config.get('use_curriculum', False):
                print("Using Curriculum Domain Randomization - Joint dropout & sensor noise!")
                print(f"  Phase 2 (0-{dr_config.get('phase_2_steps', 5000000)}): Single joint + mild noise")
                print(f"  Phase 3 ({dr_config.get('phase_2_steps', 5000000)}+): Multiple joints + high noise")
                env = CurriculumDRWrapper(env, dr_config)
            else:
                print("Using Static Domain Randomization - Joint dropout & sensor noise!")
                env = DomainRandomizationWrapper(env, dr_config)
        
        # Apply Straight-Line wrapper if specified
        if use_straight_line:
            print("Using Straight-Line Locomotion - Constraining to forward movement!")
            print(f"  Lateral penalty: {straight_line_config.get('lateral_penalty', 2.0)}")
            print(f"  Rotation penalty: {straight_line_config.get('rotation_penalty', 1.0)}")
            print(f"  Max deviation: {straight_line_config.get('max_lateral_deviation', 2.0)}m")
            from envs.straight_line_wrapper import StraightLineWrapper
            env = StraightLineWrapper(
                env,
                lateral_penalty=straight_line_config.get('lateral_penalty', 2.0),
                rotation_penalty=straight_line_config.get('rotation_penalty', 1.0),
                straight_bonus=straight_line_config.get('straight_bonus', 0.5),
                max_lateral_deviation=straight_line_config.get('max_lateral_deviation', 2.0)
            )
            
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
    use_target_walking = config.get('env', {}).get('use_target_walking', False)
    sr2l_config = config.get('sr2l', {})
    
    print(f"Experiment: {experiment_config.get('name', 'unknown')}")
    print(f"Environment: {env_name}")
    
    if use_target_walking:
        target_dist = config.get('env', {}).get('target_distance', 5.0)
        print(f"Target Walking Wrapper: ENABLED ({target_dist}m targets)")
        print("Goal: Learn to navigate to specific target positions")
    elif use_success_reward:
        print("Success Reward Wrapper: ENABLED")
        print("\nVelocity Targets:")
        print(f"  - Target velocity: 1.0 m/s (achievable for RealAnt)")
        print(f"  - Exponential speed rewards for faster walking!")
    else:
        print("Reward Wrapper: DISABLED (using default environment rewards)")
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
    
    permanent_dr_config = config.get('permanent_dr', {})
    if permanent_dr_config.get('enabled', False):
        print(f"\nPERMANENT Domain Randomization:")
        print(f"  - Status: ENABLED")
        print(f"  - Max failed joints: {permanent_dr_config.get('max_failed_joints', 4)}")
        print(f"  - Failure rate: {permanent_dr_config.get('failure_rate', 0.001)}")
        print(f"  - Curriculum duration: {permanent_dr_config.get('curriculum_steps', 10000000):,} steps")
    
    # Check for straight-line in main config (not just in create_env)
    use_straight_line = config.get('env', {}).get('use_straight_line', False)
    straight_line_config = config.get('straight_line', {})
    
    if use_straight_line:
        print(f"\nStraight-Line Locomotion:")
        print(f"  - Status: ENABLED")
        print(f"  - Lateral penalty: {straight_line_config.get('lateral_penalty', 2.0)}")
        print(f"  - Rotation penalty: {straight_line_config.get('rotation_penalty', 1.0)}")
        print(f"  - Goal: Force robot to walk straight forward (no circling!)")
    
    print(f"\nTraining:")
    print(f"  - Total timesteps: {config.get('total_timesteps', 1000000):,}")
    print(f"  - Learning rate: {config.get('ppo', {}).get('learning_rate', 0.0003)}")
    print("="*60 + "\n")
    
    # Create env config dict for the create_env function
    env_config_dict = {'env': config.get('env', {'name': env_name})}
    env = create_env(env_config_dict, normalize=True, full_config=config)
    eval_env = create_env(env_config_dict, normalize=True, norm_reward=False, full_config=config)
    
    # Define network architecture
    activation_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'leaky_relu': nn.LeakyReLU,
    }
    activation_name = config.get('policy', {}).get('activation', 'relu')
    activation_fn = activation_map.get(activation_name, nn.ReLU)
    
    policy_kwargs = dict(
        net_arch=config.get('policy', {}).get('hidden_sizes', [64, 128]),
        activation_fn=activation_fn,
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
    
    # Load from pretrained model if specified
    pretrained_path = config.get('pretrained_model')
    if pretrained_path:
        print(f"\nLoading pretrained model from: {pretrained_path}")
        if sr2l_enabled:
            # For SR2L, load the pretrained weights but keep SR2L config
            pretrained = PPO.load(pretrained_path)
            model.policy.load_state_dict(pretrained.policy.state_dict())
            print("Loaded pretrained weights into SR2L model")
        else:
            model = PPO.load(pretrained_path, env=env)
            print("Loaded pretrained PPO model")
    
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
    
    # Add custom metrics callback if using custom rewards
    if use_success_reward or use_target_walking:
        wrapper_type = "target walking" if use_target_walking else "success reward"
        print(f"Adding CustomMetricsCallback for {wrapper_type} tracking")
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