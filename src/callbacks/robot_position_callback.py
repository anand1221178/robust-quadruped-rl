#!/usr/bin/env python3
"""
Robot Position Tracking Callback
Logs robot position metrics to W&B throughout training
"""

from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional


class RobotPositionCallback(BaseCallback):
    """
    Callback to continuously log robot position metrics to W&B
    Works throughout all training phases (Phase 0, Phase 1+)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.prev_x_pos: Optional[float] = None
        self.prev_timestep = 0

    def _on_training_start(self) -> None:
        """Initialize callback at training start"""
        if self.verbose > 0:
            print("🤖 Robot Position Callback: Continuous W&B tracking enabled")

    def _on_step(self) -> bool:
        """
        Log robot position and curriculum metrics every step
        Returns: True to continue training
        """
        if self.logger is None:
            return True

        try:
            # Get the base environment to extract position and curriculum info
            if hasattr(self.model.env, 'envs') and len(self.model.env.envs) > 0:
                env = self.model.env.envs[0]

                # Check for curriculum wrapper (may be present after phase switch)
                curriculum_wrapper = None
                wrapper = env
                while hasattr(wrapper, 'env'):
                    if hasattr(wrapper, 'failed_joints') and hasattr(wrapper, 'current_phase'):
                        curriculum_wrapper = wrapper
                        break
                    wrapper = wrapper.env

                # Log curriculum metrics if systematic curriculum is active
                if curriculum_wrapper is not None:
                    self.logger.record("curriculum/current_phase", curriculum_wrapper.current_phase)
                    self.logger.record("curriculum/subphase", curriculum_wrapper.current_subphase + 1)
                    self.logger.record("curriculum/failed_joint_count", len(curriculum_wrapper.failed_joints))
                    self.logger.record("curriculum/subphase_progress", curriculum_wrapper.subphase_steps)

                    # Log pattern type as a string/category
                    pattern_type = curriculum_wrapper._get_current_pattern_type()
                    if pattern_type != 'complete':
                        self.logger.record("curriculum/pattern_type", pattern_type)
                else:
                    # Log Phase 0 (no curriculum active)
                    self.logger.record("curriculum/current_phase", 0)

                # Find the base environment for robot position
                base_env = env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env

                # Extract position from MuJoCo data
                if hasattr(base_env, 'unwrapped') and hasattr(base_env.unwrapped, 'data'):
                    x_pos = float(base_env.unwrapped.data.qpos[0])  # x position
                    z_pos = float(base_env.unwrapped.data.qpos[2])  # height

                    # Calculate velocity
                    if self.prev_x_pos is not None:
                        steps_elapsed = self.num_timesteps - self.prev_timestep
                        if steps_elapsed > 0:
                            dt = steps_elapsed * 0.05  # 50ms timesteps
                            velocity = (x_pos - self.prev_x_pos) / dt

                            # Log robot metrics to W&B using SB3's logger
                            self.logger.record("robot/x_position", x_pos)
                            self.logger.record("robot/height", z_pos)
                            self.logger.record("robot/velocity_ms", velocity)
                            self.logger.record("robot/total_distance", abs(x_pos))

                    # Update tracking variables
                    self.prev_x_pos = x_pos
                    self.prev_timestep = self.num_timesteps

        except Exception as e:
            # Don't crash training if logging fails
            if self.verbose > 0:
                print(f"Metrics logging error: {e}")

        return True