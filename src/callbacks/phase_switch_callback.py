#!/usr/bin/env python3
"""
Phase Switching Callback for Systematic Curriculum V2
Handles environment switching from pure baseline to curriculum environment
"""

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from typing import Dict


class PhaseSwitchCallback(BaseCallback):
    """
    Callback to switch from baseline environment to curriculum environment
    at specified timestep (10M steps for Phase 0 -> Phase 1 transition)
    """

    def __init__(self,
                 phase_0_duration: int,
                 config: Dict,
                 verbose: int = 0):
        super().__init__(verbose)
        self.phase_0_duration = phase_0_duration
        self.config = config
        self.switched = False
        self.current_phase = 0

    def _on_training_start(self) -> None:
        """Initialize callback at training start"""
        print(f"🎯 PHASE SWITCHING ENABLED - V2 True Phase 0")
        print(f"   Phase 0: 0 - {self.phase_0_duration:,} steps (pure baseline)")
        print(f"   Phase 1+: {self.phase_0_duration:,}+ steps (systematic curriculum)")

    def _on_step(self) -> bool:
        """
        Check if we need to switch environments
        Returns: True to continue training
        """
        # Only switch once at the phase boundary
        if not self.switched and self.num_timesteps >= self.phase_0_duration:
            self._switch_to_curriculum()
            self.switched = True

        # Log current phase and robot movement for monitoring
        if self.logger is not None:
            self.logger.record("curriculum/current_phase", self.current_phase)

            # Add robot position tracking to see if it's actually moving
            try:
                # Get the base environment to extract position
                if hasattr(self.model.env, 'envs') and len(self.model.env.envs) > 0:
                    env = self.model.env.envs[0]
                    # Unwrap to get to the RealAnt environment
                    base_env = env
                    while hasattr(base_env, 'env'):
                        base_env = base_env.env

                    # Extract position from MuJoCo data
                    if hasattr(base_env, 'unwrapped') and hasattr(base_env.unwrapped, 'data'):
                        x_pos = float(base_env.unwrapped.data.qpos[0])  # x position
                        z_pos = float(base_env.unwrapped.data.qpos[2])  # height

                        # Calculate velocity
                        if not hasattr(self, 'prev_x_pos'):
                            self.prev_x_pos = x_pos
                            self.prev_timestep = self.num_timesteps

                        steps_elapsed = self.num_timesteps - self.prev_timestep
                        if steps_elapsed > 0:
                            dt = steps_elapsed * 0.05  # 50ms timesteps
                            velocity = (x_pos - self.prev_x_pos) / dt

                            # Log robot metrics to W&B
                            self.logger.record("robot/x_position", x_pos)
                            self.logger.record("robot/height", z_pos)
                            self.logger.record("robot/velocity_ms", velocity)
                            self.logger.record("robot/total_distance", abs(x_pos))

                            # Update tracking variables
                            self.prev_x_pos = x_pos
                            self.prev_timestep = self.num_timesteps

            except Exception:
                # Don't crash training if position logging fails
                pass

        return True

    def _switch_to_curriculum(self):
        """Switch from baseline to curriculum environment"""
        print(f"\n" + "="*60)
        print(f"🔄 PHASE 0 → PHASE 1 TRANSITION AT {self.num_timesteps:,} STEPS")
        print(f"   Switching to systematic curriculum environment...")
        print("="*60)

        # Import here to avoid circular dependencies
        from envs.success_reward_wrapper import SuccessRewardWrapper
        from envs.systematic_curriculum_wrapper import SystematicCurriculumWrapper

        # Get current environment info
        num_envs = self.training_env.num_envs if hasattr(self.training_env, 'num_envs') else 1

        # Create new environment with systematic curriculum wrapper
        def make_curriculum_env():
            env_name = self.config.get('env', {}).get('name', 'RealAntMujoco-v0')
            env = gym.make(env_name)

            # Apply success reward wrapper
            if self.config.get('env', {}).get('use_success_reward', True):
                env = SuccessRewardWrapper(env)

            # Apply systematic curriculum wrapper
            curriculum_config = self.config.get('systematic_curriculum', {})
            env = SystematicCurriculumWrapper(env, curriculum_config)

            env = Monitor(env)
            return env

        # Create new vectorized environment
        new_env = DummyVecEnv([make_curriculum_env for _ in range(num_envs)])

        # Handle VecNormalize if present
        if hasattr(self.training_env, 'obs_rms'):  # Has VecNormalize
            # Create new VecNormalize with same parameters as original
            new_vec_env = VecNormalize(new_env,
                                      training=True,   # Keep training enabled
                                      norm_obs=True,
                                      norm_reward=True)

            # Copy ALL attributes from original VecNormalize
            new_vec_env.obs_rms = self.training_env.obs_rms
            new_vec_env.ret_rms = self.training_env.ret_rms
            new_vec_env.num_envs = self.training_env.num_envs
            new_vec_env.epsilon = self.training_env.epsilon
            new_vec_env.gamma = self.training_env.gamma

            # Keep training active but with existing statistics as starting point

            print("    VecNormalize statistics copied with slow adaptation enabled")

            # Update model's environment
            self.model.set_env(new_vec_env)

            # CRITICAL FIX: Force fresh rollout collection after environment switch
            # Reset PPO's internal step counters to force new rollout collection
            self.model._last_obs = None  # Force fresh observation
            self.model.num_timesteps = self.num_timesteps  # Sync timestep counters
            print("    PPO state reset for fresh rollout collection")
        else:
            # No VecNormalize, just switch environment
            self.model.set_env(new_env)

            # CRITICAL FIX: Force fresh rollout collection (no VecNormalize case)
            self.model._last_obs = None  # Force fresh observation
            self.model.num_timesteps = self.num_timesteps  # Sync timestep counters
            print("    PPO state reset for fresh rollout collection (no VecNormalize)")

        self.current_phase = 1
        print("    Environment switch complete!")
        print("    Systematic curriculum now active")
        print("="*60 + "\n")