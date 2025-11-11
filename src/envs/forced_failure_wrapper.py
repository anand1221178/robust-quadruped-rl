#!/usr/bin/env python3
"""
Forced Joint Failure Wrapper - GUARANTEED joint failures for testing

"""

import gymnasium as gym
import numpy as np
from typing import List, Optional

class ForcedFailureWrapper(gym.Wrapper):
    """
    Forces specific joints to always fail (set torque to 0)
    Used for SYSTEMATIC testing - no probabilistic dropout!
    """

    def __init__(self, env, forced_joints: List[int]):
        super().__init__(env)
        self.forced_joints = forced_joints  # List of joint indices to FORCE to fail
        self.num_joints = 8  # RealAnt has 8 joints

        print(f"FORCED FAILURE WRAPPER: Joints {self.forced_joints} will ALWAYS fail!")

    def reset(self, **kwargs):
        """Reset - forced joints are always the same"""
        obs, info = self.env.reset(**kwargs)

        # Add failure info
        if info is None:
            info = {}
        info['dropped_joints'] = self.forced_joints.copy()
        info['forced_failure'] = True

        return obs, info

    def step(self, action):
        """Apply forced joint failures to actions"""
        # Force specified joints to 0 (locked/failed)
        modified_action = action.copy()

        for joint_idx in self.forced_joints:
            if joint_idx < len(modified_action):
                modified_action[joint_idx] = 0.0

        # Take environment step with forced failures
        obs, reward, terminated, truncated, info = self.env.step(modified_action)

        # Add failure info
        if info is None:
            info = {}
        info['original_action'] = action
        info['modified_action'] = modified_action
        info['dropped_joints'] = self.forced_joints.copy()
        info['forced_failure'] = True

        return obs, reward, terminated, truncated, info