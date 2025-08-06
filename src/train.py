"""
Main training script that uses YAML configuration files
This is the proper way to handle configs!
"""

import os
import sys
import gymnasium as gym
import numpy as np
from datetime import datetime
import warnings
import yaml
from pathlib import Path
import argparse

#RL model imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

#Logging imports - just wandb
import wandb
from wandb.integration.sb3 import WandbCallback

#Supress warnings
warnings.filterwarnings("ignore", message=".*The environment Ant-v4 is out of date.*")

#Load a single config -> used for loading only PPO yaml file 
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

#merge configs funciton -> use for ablations study when we have to merge all of them together
def merge_configs(*configs):
    result = {} #this "List" will hold the configs we use
    for config in configs:
        if config:
            result.update(config)
    return result

#Create env
def create_env(env_config: dict, normalize:bool=True):
    """Create the environment based on the config passed in. Can be anything like PPO or combined (ran through merge configs)"""
    env_name = env_config['env']['name']

    def make_env():
        env = gym.make(env_name)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])

    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    return env

#Main training loop
def train(config:dict):
    """Extract configs"""
    train_config = config.get('train',{})
    env_config = config.get('env',{})
    ppo_config = config.get('ppo',{})
    policy_config = config.get('policy',{})
    logging_config = config.get('logging', {})
    experiment_config = config.get('experiment', {})

    """Initilise wandb"""
    if logging_config.get('wandb',False):
        run = wandb.init(
            project=logging_config.get('wandb_project', 'robust-quadruped-rl'),
            entity = logging_config.get('wandb_entity'),
            name = experiment_config.get('name', f"ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            config=config,
            tags=experiment_config.get('tags',['ppo']),
            sync_tensorboard=True,
        )
        run_id = run.id
    else:
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S')

    #Model Save dir
    save_path = f"experiments/{experiment_config.get('name','ppo')}_{run_id}"
    os.makedirs(save_path, exist_ok=True)

    #Save config also, to match above -. easier to track
    with open(f"{save_path}/config.yaml",'w') as f:
        yaml.dump(config,f)

    #Create env
    print(f"Creating environment: {env_config['env']['name']}")
    #Call create env func
    env = create_env(env_config,normalize=True) #Set normalise true or false here
    eval_env = create_env(env_config, normalize=False)

    #Define network architecture
    policy_kwargs = dict(
        net_arch = policy_config.get('hidden_sizes',[64,128]),
        activation_func = getattr(gym.spaces.utils.nn, policy_config.get('activation', 'ReLu').upper()),
    )

    #Create the actual PPO model with the params from the config
    print("Creating PPO model with config parameters...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 2048),
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        ent_coef=ppo_config.get('ent_coef', 0.0),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        policy_kwargs=policy_kwargs,
        verbose=logging_config.get('verbose', 1),
        tensorboard_log=f"./tensorboard/{run_id}",
        seed=train_config.get('seed', 42),
    )

    #printf config for niceness
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    print(f"Environment: {env_config['env']['name']}")
    print(f"Algorithm: PPO")
    print(f"Total timesteps: {train_config.get('total_timesteps', 1000000):,}")
    print(f"Learning rate: {ppo_config.get('learning_rate', 3e-4)}")
    print(f"Network architecture: {policy_config.get('hidden_sizes', [64, 128])}")
    print(f"Save path: {save_path}")
    print("="*60 + "\n")

