#!/usr/bin/env python3
"""
Systematic Curriculum V2 Training Script
Implements True Phase 0 approach with environment switching at 10M steps

Key Features:
- Phase 0 (0-10M): Pure baseline environment (no curriculum wrapper)
- Phase 1+ (10M+): Systematic curriculum environment with joint failures
- Smooth VecNormalize handling during transition
"""

import os
import sys
import time
import gymnasium as gym
import numpy as np
import torch
import warnings
from typing import Dict, Optional, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.envs.success_reward_wrapper import SuccessRewardWrapper
from src.envs.systematic_curriculum_wrapper import SystematicCurriculumWrapper
from src.utils.callbacks import WandbCallback

# Suppress gymnasium warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

class PhaseTransitionCallback(BaseCallback):
    """
    Callback to handle environment switching at phase transitions
    Switches from baseline environment to curriculum environment at 10M steps
    """

    def __init__(self,
                 phase_0_duration: int,
                 curriculum_config: Dict,
                 freeze_vecnorm_after_phase0: bool = True,
                 verbose: int = 0):
        super().__init__(verbose)
        self.phase_0_duration = phase_0_duration
        self.curriculum_config = curriculum_config
        self.freeze_vecnorm_after_phase0 = freeze_vecnorm_after_phase0
        self.phase_switched = False
        self.current_phase = "phase_0"

    def _on_training_start(self) -> None:
        """Called once at the beginning of training"""
        print(f"🎯 SYSTEMATIC CURRICULUM V2 - TRUE PHASE 0 TRAINING")
        print(f"   Phase 0: 0 - {self.phase_0_duration:,} steps (pure baseline)")
        print(f"   Phase 1+: {self.phase_0_duration:,}+ steps (systematic curriculum)")
        print(f"   VecNormalize freeze after Phase 0: {self.freeze_vecnorm_after_phase0}")

    def _on_step(self) -> bool:
        """
        Called at every step
        Returns: True to continue training, False to stop
        """
        # Check if we need to switch phases
        if not self.phase_switched and self.num_timesteps >= self.phase_0_duration:
            self._switch_to_curriculum_phase()

        # Log current phase to wandb if available
        if self.logger is not None:
            self.logger.record("curriculum/current_phase",
                              0 if self.current_phase == "phase_0" else 1)
            self.logger.record("curriculum/phase_name", self.current_phase)

        return True

    def _switch_to_curriculum_phase(self):
        """Handle the environment switch from baseline to curriculum"""
        print(f"\n{'='*60}")
        print(f" PHASE TRANSITION AT {self.num_timesteps:,} STEPS")
        print(f"   Switching from Phase 0 (baseline) to Phase 1 (curriculum)")
        print(f"{'='*60}\n")

        # Create new curriculum environment
        def make_curriculum_env():
            env = gym.make('RealAntMujoco-v0')
            env = Monitor(env)
            env = SuccessRewardWrapper(env)
            env = SystematicCurriculumWrapper(env, self.curriculum_config)
            return env

        # Create new vectorized environment with curriculum wrapper
        new_env = DummyVecEnv([make_curriculum_env for _ in range(self.training_env.num_envs)])

        # Transfer VecNormalize wrapper if it exists
        if hasattr(self.training_env, 'obs_rms'):  # Check if VecNormalize
            # Create new VecNormalize with same stats
            new_vec_env = VecNormalize(new_env,
                                      training=not self.freeze_vecnorm_after_phase0,
                                      norm_obs=True,
                                      norm_reward=True)

            # Copy normalization statistics
            new_vec_env.obs_rms = self.training_env.obs_rms
            new_vec_env.ret_rms = self.training_env.ret_rms

            # Freeze if requested
            if self.freeze_vecnorm_after_phase0:
                new_vec_env.training = False
                print("    VecNormalize statistics frozen")
            else:
                print("    VecNormalize continues updating")

            # Update the model's environment
            self.model.set_env(new_vec_env)
            self.training_env = new_vec_env
        else:
            # No VecNormalize, just update environment
            self.model.set_env(new_env)
            self.training_env = new_env

        self.phase_switched = True
        self.current_phase = "systematic_curriculum"

        print("    Environment switch complete!")
        print("    Systematic curriculum now active")
        print(f"{'='*60}\n")

def make_phase0_env(env_name: str = 'RealAntMujoco-v0'):
    """Create pure baseline environment for Phase 0"""
    def _init():
        env = gym.make(env_name)
        env = Monitor(env)
        env = SuccessRewardWrapper(env)
        return env
    return _init

def make_curriculum_env(env_name: str, curriculum_config: Dict):
    """Create curriculum environment for Phase 1+"""
    def _init():
        env = gym.make(env_name)
        env = Monitor(env)
        env = SuccessRewardWrapper(env)
        env = SystematicCurriculumWrapper(env, curriculum_config)
        return env
    return _init

@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def train(cfg: DictConfig):
    """
    Main training function for Systematic Curriculum V2
    Implements True Phase 0 approach with environment switching
    """

    print("="*60)
    print("🚀 SYSTEMATIC CURRICULUM V2 - TRUE PHASE 0 TRAINING")
    print("="*60)

    # Set random seeds
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Extract key configuration
    phase_0_duration = cfg.get('phase_0_duration', 10_000_000)  # 10M steps default
    curriculum_config = OmegaConf.to_container(cfg.systematic_curriculum)
    freeze_vecnorm = cfg.get('freeze_vecnorm_after_phase0', True)

    # Initialize Weights & Biases if requested
    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=f"{cfg.experiment.name}_v2",
            config=OmegaConf.to_container(cfg),
            sync_tensorboard=True
        )

    # Create Phase 0 environment (pure baseline)
    print("\nCreating Phase 0 environment (pure baseline)...")
    env = DummyVecEnv([make_phase0_env(cfg.env.name) for _ in range(cfg.num_envs)])

    # Apply VecNormalize wrapper
    if cfg.get('use_vec_normalize', True):
        # Check if we're loading pretrained VecNormalize
        if cfg.get('pretrained_vec_normalize'):
            print(f"Loading pretrained VecNormalize from {cfg.pretrained_vec_normalize}")
            env = VecNormalize.load(cfg.pretrained_vec_normalize, env)
            env.training = True  # Enable training for Phase 0
            env.norm_reward = True
        else:
            print("Creating new VecNormalize wrapper")
            env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

    # Create or load model
    if cfg.get('pretrained_model'):
        print(f"\nFINE-TUNING MODE: Loading pretrained model from {cfg.pretrained_model}")
        model = PPO.load(cfg.pretrained_model, env=env)

        # Update learning rate for fine-tuning
        model.learning_rate = cfg.ppo.learning_rate
        print(f"Updated learning rate to {cfg.ppo.learning_rate} for fine-tuning")
    else:
        print("\nCreating new PPO model from scratch")
        model = PPO(
            policy=cfg.policy.type if hasattr(cfg.policy, 'type') else 'MlpPolicy',
            env=env,
            learning_rate=cfg.ppo.learning_rate,
            n_steps=cfg.ppo.get('n_steps', 2048),
            batch_size=cfg.ppo.batch_size,
            n_epochs=cfg.ppo.n_epochs,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            clip_range=cfg.ppo.clip_range,
            ent_coef=cfg.ppo.get('ent_coef', 0.0),
            vf_coef=cfg.ppo.get('vf_coef', 0.5),
            max_grad_norm=cfg.ppo.get('max_grad_norm', 0.5),
            verbose=cfg.logging.verbose,
            tensorboard_log=f"./logs/{cfg.experiment.name}_v2" if cfg.logging.tensorboard else None,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    # Setup callbacks
    callbacks = []

    # Add phase transition callback
    phase_callback = PhaseTransitionCallback(
        phase_0_duration=phase_0_duration,
        curriculum_config=curriculum_config,
        freeze_vecnorm_after_phase0=freeze_vecnorm,
        verbose=1
    )
    callbacks.append(phase_callback)

    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoints.save_freq,
        save_path=f"./checkpoints/{cfg.experiment.name}_v2",
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)

    # Add W&B callback if enabled
    if cfg.logging.wandb:
        wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"./models/{cfg.experiment.name}_v2",
            verbose=2
        )
        callbacks.append(wandb_callback)

    # Start training
    print(f"\n🚀 Starting training for {cfg.total_timesteps:,} timesteps...")
    print(f"   Phase 0: Steps 0 - {phase_0_duration:,} (pure baseline)")
    print(f"   Phase 1+: Steps {phase_0_duration:,} - {cfg.total_timesteps:,} (systematic curriculum)")
    print("="*60)

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            log_interval=cfg.logging.log_interval,
            tb_log_name=f"{cfg.experiment.name}_v2",
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Training completed successfully
        elapsed_time = time.time() - start_time
        print(f"\n Training completed successfully!")
        print(f"   Total time: {elapsed_time/3600:.2f} hours")
        print(f"   Final timesteps: {cfg.total_timesteps:,}")

        # Save final model
        final_model_path = f"./models/{cfg.experiment.name}_v2/final_model"
        model.save(final_model_path)
        print(f"   Model saved to: {final_model_path}")

        # Save VecNormalize if used
        if cfg.get('use_vec_normalize', True):
            vec_norm_path = f"./models/{cfg.experiment.name}_v2/vec_normalize.pkl"
            env.save(vec_norm_path)
            print(f"   VecNormalize saved to: {vec_norm_path}")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise e
    finally:
        # Cleanup
        env.close()
        if cfg.logging.wandb:
            wandb.finish()

    print("="*60)
    print("🏁 SYSTEMATIC CURRICULUM V2 TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    train()