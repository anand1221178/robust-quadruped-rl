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
from envs.systematic_curriculum_wrapper import SystematicCurriculumWrapper
from envs.specialist_training_wrapper import SpecialistTrainingWrapper
from envs.v7_acdr_wrapper import V7ACDRWrapper, V7LinearCurriculumDR
from envs.v7_acdr_wrapper_fixed import V7ACDRWrapperFixed
from envs.v8_enhanced_acdr_wrapper import V8EnhancedACDRWrapper
from envs.backward_penalty_wrapper import BackwardPenaltyWrapper
from envs.rotation_mastery_wrapper import RotationMasteryWrapper

# Import RealAnt environments
import realant_sim

# Suppress warnings
warnings.filterwarnings("ignore", message=".*The environment Ant-v4 is out of date.*")


class V8ACDRWandbCallback(BaseCallback):
    """Custom callback to log V8 Enhanced ACDR metrics to W&B"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.v8_wrapper = None

    def _on_training_start(self) -> None:
        """Find the V8 wrapper in the environment stack"""
        env = self.training_env
        # Look through environment stack for V8 wrapper
        current_env = env
        while hasattr(current_env, 'envs'):
            current_env = current_env.envs[0]
        while hasattr(current_env, 'env'):
            if isinstance(current_env, V8EnhancedACDRWrapper):
                self.v8_wrapper = current_env
                print("✅ V8 Enhanced ACDR wrapper found for logging")
                break
            current_env = current_env.env

    def _on_step(self) -> bool:
        """Log V8 ACDR metrics every step"""
        if self.v8_wrapper is not None and self.num_timesteps % 1000 == 0:  # Log every 1000 steps
            try:
                # Get curriculum status
                status = self.v8_wrapper.get_curriculum_status()

                # Log to W&B
                wandb.log({
                    'v8_acdr/current_phase': status['current_phase'],
                    'v8_acdr/phase_progress': status['phase_progress'],
                    'v8_acdr/total_progress': status['total_progress'],
                    'v8_acdr/current_k_value': status['current_k_value'],
                    'v8_acdr/num_current_failures': len(status['current_failures']),
                    'v8_acdr/episode_length': status['episode_length'],
                    'v8_acdr/current_failures': str(status['current_failures']),
                    'v8_acdr/phase_description': status['phase_description']
                }, step=self.num_timesteps)

                # Also check for ankle specialization
                if hasattr(self.v8_wrapper, 'config'):
                    ankle_config = self.v8_wrapper.config.get('ankle_specialization', {})
                    if ankle_config.get('enabled', False):
                        # Check if current failures include ankles
                        ankle_joints = [1, 3, 5, 7]  # Ankle joint indices
                        ankle_failures = [j for j in status['current_failures'] if j in ankle_joints]

                        wandb.log({
                            'v8_acdr/ankle_specialization_active': len(ankle_failures) > 0,
                            'v8_acdr/ankle_failures_count': len(ankle_failures),
                            'v8_acdr/ankle_focus_ratio': ankle_config.get('phase_1_ankle_focus', 0.0)
                        }, step=self.num_timesteps)

            except Exception as e:
                if self.verbose:
                    print(f"Warning: V8 ACDR logging failed: {e}")

        return True


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with Hydra defaults support"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process Hydra defaults if present
    if 'defaults' in config:
        base_config = {}
        
        # Load each default config
        for default in config['defaults']:
            if isinstance(default, str):
                # Handle simple string defaults like "/train/default"
                default_path = default.strip('/')
                default_file = f"configs/{default_path}.yaml"
            else:
                # Handle more complex defaults (not implemented for now)
                continue
                
            if os.path.exists(default_file):
                with open(default_file, 'r') as f:
                    default_config = yaml.safe_load(f)
                    # Merge base config with default
                    base_config = {**base_config, **default_config}
        
        # Remove defaults from main config and merge with base
        main_config = {k: v for k, v in config.items() if k != 'defaults'}
        
        # Deep merge base_config with main_config (main_config overrides)
        final_config = deep_merge(base_config, main_config)
        return final_config
    
    return config


def deep_merge(base_dict: dict, override_dict: dict) -> dict:
    """Deep merge two dictionaries, with override_dict taking precedence"""
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def create_env(config: dict, normalize: bool = True, norm_reward: bool = True):
    """Create environment - SIMPLIFIED for forward locomotion"""
    env_name = config.get('env', {}).get('name', 'RealAntMujoco-v0')
    
    # Simple options
    use_success_reward = config.get('env', {}).get('use_success_reward', False)
    use_domain_randomization = config.get('env', {}).get('use_domain_randomization', False)
    use_specialist_wrapper = config.get('env', {}).get('use_specialist_wrapper', False)
    use_v7_acdr = config.get('env', {}).get('use_v7_acdr', False)
    use_v7_acdr_fixed = config.get('env', {}).get('use_v7_acdr_fixed', False)

    def make_env():
        env = gym.make(env_name)

        # Apply reward wrapper
        if use_success_reward:
            print("✅ Success Reward Wrapper: Forward locomotion training")
            env = SuccessRewardWrapper(env)

        # V7.6: Apply backward penalty if configured
        if config.get('env', {}).get('use_backward_penalty', False):
            penalty_mult = config.get('env', {}).get('backward_penalty_multiplier', 5.0)
            stationary_penalty = config.get('env', {}).get('stationary_penalty', 0.0)
            stationary_threshold = config.get('env', {}).get('stationary_threshold', 0.05)
            env = BackwardPenaltyWrapper(env,
                                        penalty_multiplier=penalty_mult,
                                        stationary_penalty=stationary_penalty,
                                        stationary_threshold=stationary_threshold)

        # V7.11: Apply rotation mastery wrapper if configured
        if config.get('env', {}).get('use_rotation_mastery', False):
            rotation_config = config.get('env', {}).get('rotation_mastery', {})
            print("🎯 Rotation Mastery Wrapper: Teaching ankle_4 rotation strategy")
            env = RotationMasteryWrapper(env, rotation_config)

        # V7_FIXED: The new, corrected ACDR wrapper
        if use_v7_acdr_fixed:
            acdr_config = config.get('v7_acdr', {})
            print(f"🔥🔥🔥 V7 ACDR FIXED: Using CORRECTED wrapper with raw rewards!")
            env = V7ACDRWrapperFixed(
                env,
                **acdr_config
            )
            return Monitor(env)

        # V7: ACDR Hard2Easy Curriculum (Research-validated approach)
        if use_v7_acdr:
            acdr_config = config.get('v7_acdr', {})
            curriculum_type = acdr_config.get('curriculum_type', 'hard2easy')

            print(f"🔥 V7 ACDR CURRICULUM: {curriculum_type.upper()}")
            if curriculum_type == 'hard2easy':
                print("   ✅ Using PROVEN hard2easy approach from ACDR paper")
                print("   Starting with k=0 (dead joints) → k=1.5 (mild failures)")
            else:
                print("   ⚠️ WARNING: Easy2hard shown to fail in research!")
                print("   Starting with k=1.5 (mild failures) → k=0 (dead joints)")

            env = V7ACDRWrapper(
                env,
                curriculum_type=curriculum_type,
                initial_L=acdr_config.get('initial_L', 0.0),
                initial_U=acdr_config.get('initial_U', 0.0),
                target_L=acdr_config.get('target_L', 1.0),
                target_U=acdr_config.get('target_U', 1.5),
                update_step=acdr_config.get('update_step', 0.01),
                performance_window=acdr_config.get('performance_window', 100),
                performance_threshold=acdr_config.get('performance_threshold', None),
                # V7.2 Dual-Phase Parameters
                phase_1_target=acdr_config.get('phase_1_target', None),
                phase_2_minimum=acdr_config.get('phase_2_minimum', None),
                # V7.3 Multi-Objective Parameters
                speed_weight=acdr_config.get('speed_weight', 0.7),
                robustness_weight=acdr_config.get('robustness_weight', 0.3),
                adaptive_weights=acdr_config.get('adaptive_weights', False),
                rewind_threshold=acdr_config.get('rewind_threshold', None),
                rewind_steps=acdr_config.get('rewind_steps', 2),
                consolidation_period=acdr_config.get('consolidation_period', 1000000),
                adaptive_curriculum=acdr_config.get('adaptive_curriculum', False),
                verbose=acdr_config.get('verbose', True)
            )
            return Monitor(env)  # Early return for V7 ACDR

        # V6: Specialist training wrapper
        if use_specialist_wrapper:
            specialist_type = config.get('env', {}).get('specialist_type', 'normal')
            training_phase = config.get('env', {}).get('training_phase', 'baseline')
            print(f"🎯 V6 SPECIALIST TRAINING: {specialist_type}")
            print(f"   Training Phase: {training_phase}")
            env = SpecialistTrainingWrapper(env, specialist_type, training_phase)
            return Monitor(env)  # Early return for specialist training
        
        # Check if using phase switching (V2 approach)
        use_phase_switching = config.get('phase_switching', {}).get('enabled', False)

        # Apply systematic curriculum or domain randomization
        if use_phase_switching:
            # V2: True Phase 0 - Don't apply curriculum wrapper yet
            print("🚀 SYSTEMATIC CURRICULUM V2: True Phase 0 enabled!")
            print(f"  Phase 0: Pure baseline environment (no curriculum wrapper)")
            print(f"  Curriculum will activate via callback at phase transition")
            # Skip applying SystematicCurriculumWrapper here

        elif config.get('systematic_curriculum', {}).get('enabled', False):
            # V1: Original approach - apply curriculum from start
            print("🎯 SYSTEMATIC CURRICULUM: Guaranteed joint failure robustness! 🎯")
            curriculum_config = config.get('systematic_curriculum', {})
            print(f"  Training phases: Single→Dual→Triple joint failures")
            print(f"  Single joint duration: {curriculum_config.get('single_joint_duration', 3000000):,} steps each")
            print(f"  Dual combo duration: {curriculum_config.get('dual_combo_duration', 3000000):,} steps each")
            env = SystematicCurriculumWrapper(env, curriculum_config)
            
        elif use_domain_randomization:
            # Traditional probabilistic domain randomization
            dr_config = config.get('domain_randomization', {})

            # 🔥 CHECK WRAPPER TYPE PREFERENCE FROM CONFIG (V8 fix: check env section first)
            wrapper_type = config.get('env', {}).get('wrapper_type', dr_config.get('wrapper_type', 'auto'))
            has_curriculum = any(key.startswith('phase_') for key in dr_config.keys())
            use_curriculum = dr_config.get('use_curriculum', True)
            
            # Determine which wrapper to use
            if wrapper_type == 'V8EnhancedACDRWrapper':
                print("🚀 V8 ENHANCED ACDR: Adaptation-Focused Learning! 🚀")
                print("🎯 Extended episodes, ankle specialization, dynamic failures")
                v8_config = config.get('env', {}).get('v8_enhanced_acdr', {})
                env = V8EnhancedACDRWrapper(env, v8_config)
            elif wrapper_type == 'CurriculumDRWrapper' or (wrapper_type == 'auto' and has_curriculum and use_curriculum):
                print("🔥 CURRICULUM DOMAIN RANDOMIZATION 🔥")

                # Print detailed curriculum schedule
                print("   📋 Curriculum Schedule:")

                # Phase 1
                phase_1_steps = dr_config.get('phase_1_steps', 0)
                phase_1_prob = dr_config.get('phase_1_config', {}).get('joint_dropout_prob', 0)
                phase_1_max = dr_config.get('phase_1_config', {}).get('max_dropped_joints', 0)
                print(f"   📍 Phase 1 (0-{phase_1_steps/1e6:.0f}M): {phase_1_prob*100:.0f}% failure rate, max {phase_1_max} joints")

                # Phase 2
                phase_2_steps = dr_config.get('phase_2_steps', 0)
                phase_2_prob = dr_config.get('phase_2_config', {}).get('joint_dropout_prob', 0)
                phase_2_min = dr_config.get('phase_2_config', {}).get('min_dropped_joints', 0)
                phase_2_max = dr_config.get('phase_2_config', {}).get('max_dropped_joints', 0)
                total_2 = phase_1_steps + phase_2_steps
                print(f"   📍 Phase 2 ({phase_1_steps/1e6:.0f}-{total_2/1e6:.0f}M): {phase_2_prob*100:.0f}% failure rate, {phase_2_min}-{phase_2_max} joints")

                # Phase 3 (if exists)
                phase_3_steps = dr_config.get('phase_3_steps', 0)
                if phase_3_steps > 0:
                    phase_3_prob = dr_config.get('phase_3_config', {}).get('joint_dropout_prob', 0)
                    phase_3_min = dr_config.get('phase_3_config', {}).get('min_dropped_joints', 0)
                    phase_3_max = dr_config.get('phase_3_config', {}).get('max_dropped_joints', 0)
                    total_3 = total_2 + phase_3_steps
                    print(f"   📍 Phase 3 ({total_2/1e6:.0f}-{total_3/1e6:.0f}M): {phase_3_prob*100:.0f}% failure rate, {phase_3_min}-{phase_3_max} joints")

                env = CurriculumDRWrapper(env, config) # Pass full config
            elif wrapper_type == 'DomainRandomizationWrapper' or (wrapper_type == 'auto' and not use_curriculum):
                print("🎲 Basic Domain Randomization: Joint failure robustness")
                print(f"  Joint dropout prob: {dr_config.get('joint_dropout_prob', 0.1)}")
                print(f"  Max dropped joints: {dr_config.get('max_dropped_joints', 2)}")
                env = DomainRandomizationWrapper(env, dr_config)
            else:
                # Default behavior - auto-detect based on phases
                if has_curriculum:
                    print("🔥 CURRICULUM DOMAIN RANDOMIZATION 🔥")
                    print("📚 Phase-based training detected - using CurriculumDRWrapper")

                    # Print detailed curriculum schedule
                    print("   📋 Curriculum Schedule:")

                    # Phase 1
                    phase_1_steps = dr_config.get('phase_1_steps', 0)
                    phase_1_prob = dr_config.get('phase_1_config', {}).get('joint_dropout_prob', 0)
                    phase_1_max = dr_config.get('phase_1_config', {}).get('max_dropped_joints', 0)
                    print(f"   📍 Phase 1 (0-{phase_1_steps/1e6:.0f}M): {phase_1_prob*100:.0f}% failure rate, max {phase_1_max} joints")

                    # Phase 2
                    phase_2_steps = dr_config.get('phase_2_steps', 0)
                    phase_2_prob = dr_config.get('phase_2_config', {}).get('joint_dropout_prob', 0)
                    phase_2_min = dr_config.get('phase_2_config', {}).get('min_dropped_joints', 0)
                    phase_2_max = dr_config.get('phase_2_config', {}).get('max_dropped_joints', 0)
                    total_2 = phase_1_steps + phase_2_steps
                    print(f"   📍 Phase 2 ({phase_1_steps/1e6:.0f}-{total_2/1e6:.0f}M): {phase_2_prob*100:.0f}% failure rate, {phase_2_min}-{phase_2_max} joints")

                    # Phase 3 (if exists)
                    phase_3_steps = dr_config.get('phase_3_steps', 0)
                    if phase_3_steps > 0:
                        phase_3_prob = dr_config.get('phase_3_config', {}).get('joint_dropout_prob', 0)
                        phase_3_min = dr_config.get('phase_3_config', {}).get('min_dropped_joints', 0)
                        phase_3_max = dr_config.get('phase_3_config', {}).get('max_dropped_joints', 0)
                        total_3 = total_2 + phase_3_steps
                        print(f"   📍 Phase 3 ({total_2/1e6:.0f}-{total_3/1e6:.0f}M): {phase_3_prob*100:.0f}% failure rate, {phase_3_min}-{phase_3_max} joints")

                    env = CurriculumDRWrapper(env, dr_config)
                else:
                    print("🎲 Basic Domain Randomization: Joint failure robustness")
                    print(f"  Joint dropout prob: {dr_config.get('joint_dropout_prob', 0.1)}")
                    print(f"  Max dropped joints: {dr_config.get('max_dropped_joints', 2)}")
                    env = DomainRandomizationWrapper(env, dr_config)
            
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])

    # 🔥 V3/V4 FIX: Read VecNormalize config from config file
    if normalize:
        vec_norm_config = config.get('vec_normalize', {})

        # Use config values or sensible defaults
        norm_obs = vec_norm_config.get('norm_obs', True)
        norm_reward_config = vec_norm_config.get('norm_reward', norm_reward)  # Config overrides parameter
        clip_obs = vec_norm_config.get('clip_obs', 10.0)

        print(f"🔧 VecNormalize Config:")
        print(f"  norm_obs: {norm_obs}")
        print(f"  norm_reward: {norm_reward_config} {'🔥 FIXED for V3/V4!' if not norm_reward_config else ''}")
        print(f"  clip_obs: {clip_obs}")

        env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward_config, clip_obs=clip_obs)

        # V5: Connect VecNormalize to SystematicCurriculumWrapper for smart reward stats reset
        if hasattr(env.venv.envs[0], 'env') and hasattr(env.venv.envs[0].env, 'set_vecnormalize_env'):
            # Navigate through DummyVecEnv -> Monitor -> SystematicCurriculumWrapper
            wrapper_env = env.venv.envs[0].env  # Monitor wrapper
            if hasattr(wrapper_env, 'env') and hasattr(wrapper_env.env, 'set_vecnormalize_env'):
                curriculum_wrapper = wrapper_env.env  # SystematicCurriculumWrapper
                curriculum_wrapper.set_vecnormalize_env(env)
                print(f"🔗 V5: VecNormalize connected to SystematicCurriculumWrapper")

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

    # Check if VecNormalize should be enabled (default: True for backward compatibility)
    vec_normalize_config = config.get('vec_normalize', {})
    use_vecnormalize = vec_normalize_config.get('enabled', True)

    print(f"VecNormalize: {'✅ ENABLED' if use_vecnormalize else '❌ DISABLED (ABLATION)'}")

    # Create environment
    env = create_env(config, normalize=use_vecnormalize, norm_reward=True)
    
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

    # Robot position tracking callback (continuous W&B logging)
    from callbacks.robot_position_callback import RobotPositionCallback
    robot_callback = RobotPositionCallback(verbose=1)
    callbacks.append(robot_callback)
    print("✅ Robot position callback added (continuous W&B tracking)")

    # Curriculum phase logging callback (for DR models with curriculum)
    if config.get('env', {}).get('use_domain_randomization', False):
        dr_config = config.get('domain_randomization', {})
        # Check if using curriculum DR (has phase configs)
        has_curriculum = 'phase_1_config' in dr_config or 'phase_1_steps' in dr_config
        if has_curriculum:
            from callbacks.curriculum_logging_callback import CurriculumLoggingCallback
            curriculum_callback = CurriculumLoggingCallback(verbose=1)
            callbacks.append(curriculum_callback)
            print("✅ Curriculum phase logging callback added (tracks DR phases to W&B)")

    # Phase switching callback for V2 (if enabled)
    if config.get('phase_switching', {}).get('enabled', False):
        from callbacks.phase_switch_callback import PhaseSwitchCallback
        phase_0_duration = config.get('phase_switching', {}).get('phase_0_duration', 10000000)
        phase_callback = PhaseSwitchCallback(
            phase_0_duration=phase_0_duration,
            config=config,
            verbose=1
        )
        callbacks.append(phase_callback)
        print(f"✅ Phase switching callback added (transition at {phase_0_duration:,} steps)")
    
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

        # Add V8 Enhanced ACDR logging callback if using V8 wrapper
        wrapper_type = config.get('env', {}).get('wrapper_type', 'auto')
        if wrapper_type == 'V8EnhancedACDRWrapper':
            v8_callback = V8ACDRWandbCallback(verbose=1)
            callbacks.append(v8_callback)
            print("✅ V8 Enhanced ACDR W&B logging callback added")
    
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