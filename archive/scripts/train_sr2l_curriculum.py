#!/usr/bin/env python3
"""
Curriculum training for SR2L
Stage 1: Learn to walk (no SR2L)
Stage 2: Add gentle SR2L
Stage 3: Increase SR2L strength
"""

import os
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from agents.ppo_sr2l import PPO_SR2L
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def create_env():
    """Create environment with success wrapper"""
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    return env

def train_curriculum():
    """Train SR2L with curriculum learning"""
    
    save_path = "experiments/ppo_sr2l_curriculum"
    os.makedirs(save_path, exist_ok=True)
    
    # Create environment
    env = create_env()
    eval_env = create_env()
    
    # Stage 1: Train WITHOUT SR2L first (learn to walk)
    print("="*60)
    print("STAGE 1: Learning to walk (No SR2L)")
    print("="*60)
    
    sr2l_config_stage1 = {
        'enabled': False  # Completely disabled
    }
    
    model = PPO_SR2L(
        "MlpPolicy",
        env,
        sr2l_config=sr2l_config_stage1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=f"{save_path}/tensorboard",
    )
    
    # Train for 2M steps without SR2L
    model.learn(total_timesteps=2000000, progress_bar=True)
    model.save(f"{save_path}/stage1_model")
    print("Stage 1 complete - Robot should be walking now")
    
    # Stage 2: Add VERY gentle SR2L
    print("\n" + "="*60)
    print("STAGE 2: Adding gentle SR2L")
    print("="*60)
    
    # Update SR2L config
    model.sr2l_config = {
        'enabled': True,
        'lambda': 0.0001,  # Extremely gentle
        'perturbation_std': 0.005,  # Tiny noise
        'apply_frequency': 20,  # Only every 20 updates
        'warmup_steps': 0,  # No warmup needed, already trained
        'max_perturbation': 0.01,
        'log_smoothness_metrics': True
    }
    
    # Continue training for 2M more steps
    model.learn(total_timesteps=2000000, progress_bar=True, reset_num_timesteps=False)
    model.save(f"{save_path}/stage2_model")
    print("Stage 2 complete - SR2L gently applied")
    
    # Stage 3: Increase SR2L strength
    print("\n" + "="*60)
    print("STAGE 3: Increasing SR2L strength")
    print("="*60)
    
    # Stronger SR2L
    model.sr2l_config['lambda'] = 0.001
    model.sr2l_config['perturbation_std'] = 0.02
    model.sr2l_config['apply_frequency'] = 5
    
    # Final training
    model.learn(total_timesteps=2000000, progress_bar=True, reset_num_timesteps=False)
    model.save(f"{save_path}/final_model")
    env.save(f"{save_path}/vec_normalize.pkl")
    
    print("\n" + "="*60)
    print("CURRICULUM TRAINING COMPLETE!")
    print("="*60)
    print(f"Models saved to: {save_path}/")
    print("- stage1_model.zip: Walking without SR2L")
    print("- stage2_model.zip: Walking with gentle SR2L")  
    print("- final_model.zip: Walking with full SR2L")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train_curriculum()