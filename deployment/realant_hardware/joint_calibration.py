#!/usr/bin/env python3
"""
Joint Calibration for RealANT Simulation-to-Hardware Transfer

This module provides the proper mapping between MuJoCo simulation joint angles
and Dynamixel servo positions for the physical RealANT robot.

Author: Based on supervisor feedback - proper calibration needed!
"""

import numpy as np

class RealAntJointCalibration:
    """
    Handles conversion between MuJoCo joint angles and Dynamixel positions.

    MuJoCo uses degrees with different ranges/centers per joint.
    Dynamixel uses positions [0-1023] where 512 = mechanical center.
    """

    def __init__(self):
        # Dynamixel AX-12A specs
        self.DYNAMIXEL_CENTER = 512
        self.DYNAMIXEL_MIN_SAFE = 224
        self.DYNAMIXEL_MAX_SAFE = 800
        self.DEGREES_PER_POSITION = 300.0 / 1024.0  # AX-12A: 300° total range
        self.POSITIONS_PER_DEGREE = 1024.0 / 300.0  # ~3.41 positions per degree

        # MuJoCo joint specifications (from mujoco.xml)
        # Joint order: [hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4]

        self.joint_names = [
            'hip_1', 'ankle_1',    # Front left leg
            'hip_2', 'ankle_2',    # Front right leg
            'hip_3', 'ankle_3',    # Back left leg
            'hip_4', 'ankle_4'     # Back right leg
        ]

        # MuJoCo ranges in degrees
        self.mujoco_ranges = np.array([
            [-40, 40],    # hip_1
            [30, 100],    # ankle_1
            [-40, 40],    # hip_2
            [-100, -30],  # ankle_2 (inverted!)
            [-40, 40],    # hip_3
            [-100, -30],  # ankle_3 (inverted!)
            [-40, 40],    # hip_4
            [30, 100]     # ankle_4
        ])

        # Calculate center position for each joint in MuJoCo
        self.mujoco_centers = np.mean(self.mujoco_ranges, axis=1)
        # Result: [0, 65, 0, -65, 0, -65, 0, 65]

        # Calculate range span for each joint
        self.mujoco_spans = np.diff(self.mujoco_ranges, axis=1).flatten()
        # Result: [80, 70, 80, 70, 80, 70, 80, 70]

        print(f"Joint Calibration Initialized:")
        print(f"  MuJoCo centers (°): {self.mujoco_centers}")
        print(f"  MuJoCo spans (°):   {self.mujoco_spans}")

    def policy_action_to_mujoco_degrees(self, policy_actions: np.ndarray) -> np.ndarray:
        """
        Convert policy actions [-1, 1] to MuJoCo joint angles in degrees.

        The policy outputs normalized actions in [-1, 1] range.
        These need to be mapped to actual joint angles based on each joint's range.

        CRITICAL: In mujoco.py, setpoints are multiplied by 2 (line 65),
        so we need to scale down by 50% to compensate!

        Args:
            policy_actions: 8D array of policy outputs in [-1, 1]

        Returns:
            8D array of joint angles in degrees
        """
        # Map [-1, 1] to each joint's specific range
        # action = -1 → min angle, action = 0 → center, action = +1 → max angle

        # CRITICAL: Scale appropriately because MuJoCo multiplies by 2!
        # 0.7 scaling - balance between movement size and safety
        # Combined with 15Hz mirror frequency to reduce communication load
        half_spans = self.mujoco_spans / 2.0
        mujoco_angles = self.mujoco_centers + policy_actions * half_spans * 0.7  # 70% scaling!

        return mujoco_angles

    def mujoco_degrees_to_dynamixel(self, mujoco_angles: np.ndarray) -> np.ndarray:
        """
        Convert MuJoCo joint angles (degrees) to Dynamixel positions.

        This is the critical calibration step!

        Args:
            mujoco_angles: 8D array of joint angles in degrees

        Returns:
            8D array of Dynamixel positions [0-1023]
        """
        # Convert degrees to Dynamixel positions
        # Note: MuJoCo center angles may not align with Dynamixel mechanical center (512)
        # For now, assume they do align (can be tuned with real robot testing)

        # Convert angle difference from center to position difference
        angle_from_center = mujoco_angles - self.mujoco_centers
        position_offset = angle_from_center * self.POSITIONS_PER_DEGREE

        dynamixel_positions = self.DYNAMIXEL_CENTER + position_offset

        # Apply safety limits
        dynamixel_positions = np.clip(
            dynamixel_positions,
            self.DYNAMIXEL_MIN_SAFE,
            self.DYNAMIXEL_MAX_SAFE
        )

        return dynamixel_positions.astype(int)

    def policy_action_to_dynamixel(
        self,
        policy_actions: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Complete pipeline: policy actions → Dynamixel positions.

        Args:
            policy_actions: 8D array of policy outputs in [-1, 1]
            verbose: Print debug info

        Returns:
            8D array of Dynamixel positions [224-800]
        """
        # Step 1: Actions to MuJoCo angles
        mujoco_angles = self.policy_action_to_mujoco_degrees(policy_actions)

        # Step 2: MuJoCo angles to Dynamixel positions
        dynamixel_positions = self.mujoco_degrees_to_dynamixel(mujoco_angles)

        if verbose:
            print(f"\nJoint Mapping:")
            for i, name in enumerate(self.joint_names):
                print(f"  {name:8s}: action={policy_actions[i]:+.2f} → "
                      f"angle={mujoco_angles[i]:+6.1f}° → "
                      f"position={dynamixel_positions[i]:4d}")

        return dynamixel_positions


# Test the calibration
if __name__ == "__main__":
    print("="*70)
    print("RealANT Joint Calibration Test")
    print("="*70)

    calibration = RealAntJointCalibration()

    # Test with sample policy outputs
    print("\n" + "="*70)
    print("Test 1: All joints at center (action = 0)")
    print("="*70)
    test_actions = np.zeros(8)
    positions = calibration.policy_action_to_dynamixel(test_actions, verbose=True)
    print(f"\nResult: All positions should be near 512 (center)")
    print(f"Positions: {positions}")

    print("\n" + "="*70)
    print("Test 2: All joints at maximum (action = +1)")
    print("="*70)
    test_actions = np.ones(8)
    positions = calibration.policy_action_to_dynamixel(test_actions, verbose=True)

    print("\n" + "="*70)
    print("Test 3: All joints at minimum (action = -1)")
    print("="*70)
    test_actions = -np.ones(8)
    positions = calibration.policy_action_to_dynamixel(test_actions, verbose=True)

    print("\n" + "="*70)
    print("Test 4: Walking pattern (alternating legs)")
    print("="*70)
    test_actions = np.array([-1, -1, 1, 1, 1, -1, -1, 1])
    positions = calibration.policy_action_to_dynamixel(test_actions, verbose=True)

    print("\n✅ Calibration module ready!")
