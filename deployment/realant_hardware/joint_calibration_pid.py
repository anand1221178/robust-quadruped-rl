#!/usr/bin/env python3
"""
Joint Calibration with PID Smoothing for RealANT

CRITICAL DISCOVERY: The policy outputs BINARY actions (50% at -1, 47% at +1).
This works in MuJoCo because the PID controller smooths the commands.
On the real robot WITHOUT PID, servos jump violently between extremes!

This module implements the SAME PID controller as MuJoCo to smooth binary actions.
"""

import numpy as np

class RealAntJointCalibrationPID:
    """
    Joint calibration WITH PID smoothing (mimics MuJoCo's controller).

    **Key insight from diagnostics:**
    - Policy uses bang-bang control (binary: ±1)
    - MuJoCo's PID smooths this into coordinated walking
    - Real robot needs same PID to avoid violent thrashing!
    """

    # Dynamixel AX-12A specs
    POSITIONS_PER_DEGREE = 1024 / 300.0  # ~3.41 positions per degree
    SAFE_MIN = 224
    SAFE_MAX = 800

    def __init__(self, dt: float = 0.02, verbose: bool = True, max_step_size: int = 50):
        """
        Args:
            dt: Control timestep (default 0.02s = 50Hz like MuJoCo)
            verbose: Print initialization info
            max_step_size: Max Dynamixel position change per command (SAFETY!)
        """
        self.dt = dt
        self.max_step_size = max_step_size  # Prevent violent jumps!

        # PID parameters tuned for real servos (not MuJoCo physics!)
        # More aggressive than MuJoCo since we need faster convergence for position control
        self.Kp = 1.0  # Reduced from 2.0 for smoother response
        self.Ki = 0.0  # No integral term
        self.Kd = 0.5  # Reduced from 1.0 - less oscillation

        # Track last commanded positions for rate limiting
        self.last_commanded_positions = None

        # MuJoCo joint specifications (from mujoco.xml)
        self.joint_names = [
            'hip_1', 'ankle_1', 'hip_2', 'ankle_2',
            'hip_3', 'ankle_3', 'hip_4', 'ankle_4'
        ]

        # MuJoCo ranges in degrees [min, max] per joint
        # CRITICAL FIX: Left ankles (s2, s8) have OPPOSITE centers from MuJoCo XML!
        # MuJoCo XML says ankle_1=65°, ankle_4=65°, but real robot needs -65° for left ankles!
        self.mujoco_ranges = np.array([
            [-40, 40],    # hip_1 (FL hip)
            [-100, -30],  # ankle_1 (FL ankle) - FLIPPED from [30,100]!
            [-40, 40],    # hip_2 (FR hip)
            [-100, -30],  # ankle_2 (FR ankle)
            [-40, 40],    # hip_3 (BR hip)
            [-100, -30],  # ankle_3 (BR ankle)
            [-40, 40],    # hip_4 (BL hip)
            [-100, -30]   # ankle_4 (BL ankle) - FLIPPED from [30,100]!
        ])

        # Calculate center position for each joint
        self.mujoco_centers = np.mean(self.mujoco_ranges, axis=1)
        # Result: [0, -65, 0, -65, 0, -65, 0, -65] degrees
        # ALL ankles now have center = -65° (toes down!)

        # Calculate range span for each joint
        self.mujoco_spans = np.diff(self.mujoco_ranges, axis=1).flatten()
        # Result: [80, 70, 80, 70, 80, 70, 80, 70] degrees

        # No inversion flags needed! Fixed the root cause (center angles) instead.
        # All joints use normal mapping now.
        self.invert_flags = np.ones(8)

        # Dynamixel ranges for each joint type
        # CRITICAL: Hips and ankles have DIFFERENT Dynamixel ranges!
        # - Hips: roughly centered around 512
        # - Ankles: range from 224 (standing/extended) to 800 (flexed)
        self.dynamixel_ranges = np.array([
            [412, 612],   # hip_1 (FL hip): 512 ± 100
            [224, 800],   # ankle_1 (FL ankle): standing to flexed
            [412, 612],   # hip_2 (FR hip): 512 ± 100
            [224, 800],   # ankle_2 (FR ankle): standing to flexed
            [412, 612],   # hip_3 (BL hip): 512 ± 100
            [224, 800],   # ankle_3 (BL ankle): standing to flexed
            [412, 612],   # hip_4 (BR hip): 512 ± 100
            [224, 800]    # ankle_4 (BR ankle): standing to flexed
        ])

        # Calculate Dynamixel center for each joint
        self.dynamixel_centers = np.mean(self.dynamixel_ranges, axis=1)
        # Result: [512, 512, 512, 512, 512, 512, 512, 512] for hips
        #         [512, 512, 512, 512] for ankles (center of [224, 800])

        # Calculate Dynamixel span for each joint
        self.dynamixel_spans = np.diff(self.dynamixel_ranges, axis=1).flatten()
        # Result: [200, 576, 200, 576, 200, 576, 200, 576]

        # PID state tracking (in RADIANS like MuJoCo uses internally)
        self.current_positions_rad = np.deg2rad(self.mujoco_centers)  # Start at center
        self.target_positions_rad = np.deg2rad(self.mujoco_centers)
        self.past_errors = np.zeros(8)

        if verbose:
            print(f"Joint Calibration with PID Initialized:")
            print(f"  MuJoCo centers (°): {self.mujoco_centers}")
            print(f"  MuJoCo spans (°):   {self.mujoco_spans}")
            print(f"  Dynamixel centers:  {self.dynamixel_centers.astype(int)}")
            print(f"  Dynamixel spans:    {self.dynamixel_spans.astype(int)}")
            print(f"  PID params: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}")
            print(f"  Control frequency: {1/dt:.0f}Hz")
            print(f"  Max step size: {self.max_step_size} Dynamixel units (safety)")

    def policy_action_to_target_radians(self, policy_actions: np.ndarray) -> np.ndarray:
        """
        Convert policy actions [-1, 1] to target joint angles in RADIANS.

        CRITICAL: Mimics MuJoCo's behavior exactly (mujoco.py line 65):
            err = 2 * setpoints - joint_positions

        This means MuJoCo multiplies policy actions by 2!

        Args:
            policy_actions: 8D array in [-1, 1]

        Returns:
            8D array of target angles in RADIANS
        """
        # CRITICAL: Apply inversion flags for left ankles FIRST!
        # This flips the sign of actions for inverted joints
        inverted_actions = policy_actions * self.invert_flags

        # Map [-1, 1] to each joint's specific range IN DEGREES
        half_spans = self.mujoco_spans / 2.0
        target_angles_deg = self.mujoco_centers + inverted_actions * half_spans

        # Convert to radians
        target_angles_rad = np.deg2rad(target_angles_deg)

        # CRITICAL: Apply MuJoCo's 2x multiplication!
        # The policy was TRAINED with this - expects actions to be doubled!
        target_angles_rad = 2.0 * target_angles_rad

        return target_angles_rad

    def update_pid(self, target_positions_rad: np.ndarray) -> np.ndarray:
        """
        Update PID controller state and compute smoothed output positions.

        This mimics MuJoCo's PID controller (mujoco.py lines 65-72).

        Args:
            target_positions_rad: 8D array of target angles in radians

        Returns:
            8D array of ACTUAL positions in radians (smoothed by PID)
        """
        # Store targets
        self.target_positions_rad = target_positions_rad

        # Compute error (like MuJoCo line 65, but we already did the 2x multiplication)
        errors = target_positions_rad - self.current_positions_rad

        # Derivative term
        d_errors = errors - self.past_errors

        # PID output (mujoco.py lines 70-72 compute torques, but we directly compute position change)
        # For position control on real servos, we'll use PID to smooth the position commands
        position_change = (
            self.Kp * errors +
            self.Ki * 0 +  # No integral term
            self.Kd * d_errors
        ) * self.dt

        # Update current positions (smooth movement toward target)
        self.current_positions_rad += position_change

        # Store past errors for next iteration
        self.past_errors = errors.copy()

        return self.current_positions_rad

    def radians_to_dynamixel(self, joint_positions_rad: np.ndarray, rate_limit: bool = True) -> np.ndarray:
        """
        Convert joint angles in radians to Dynamixel servo positions.

        Args:
            joint_positions_rad: 8D array in radians
            rate_limit: Apply rate limiting to prevent violent jumps (SAFETY!)

        Returns:
            8D array of Dynamixel positions [224-800]
        """
        # Convert radians to degrees
        joint_angles_deg = np.rad2deg(joint_positions_rad)

        # Map MuJoCo angles to Dynamixel positions using per-joint ranges
        # For each joint: normalize to [0,1] within MuJoCo range, then map to Dynamixel range

        # Normalize MuJoCo angle to [0, 1] within each joint's range
        normalized = (joint_angles_deg - self.mujoco_ranges[:, 0]) / self.mujoco_spans

        # Map to Dynamixel range for each joint
        dynamixel_positions = (
            self.dynamixel_ranges[:, 0] +
            normalized * self.dynamixel_spans
        )

        # Apply safety limits
        dynamixel_positions = np.clip(
            dynamixel_positions,
            self.SAFE_MIN,
            self.SAFE_MAX
        )

        # CRITICAL SAFETY: Rate limiting to prevent violent jumps!
        if rate_limit and self.last_commanded_positions is not None:
            # Limit change per step
            delta = dynamixel_positions - self.last_commanded_positions
            delta = np.clip(delta, -self.max_step_size, self.max_step_size)
            dynamixel_positions = self.last_commanded_positions + delta

        # Store for next iteration
        self.last_commanded_positions = dynamixel_positions.copy()

        return dynamixel_positions.astype(int)

    def policy_action_to_dynamixel(
        self,
        policy_actions: np.ndarray,
        use_pid: bool = True,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Complete pipeline with PID smoothing: policy actions → Dynamixel positions.

        Args:
            policy_actions: 8D array in [-1, 1] (can be binary!)
            use_pid: Apply PID smoothing (CRITICAL for binary actions!)
            verbose: Print debug info

        Returns:
            8D array of Dynamixel positions [224-800]
        """
        # Step 1: Actions to target angles in radians (with 2x multiplication)
        target_angles_rad = self.policy_action_to_target_radians(policy_actions)

        # Step 2: Apply PID smoothing (THIS IS THE KEY!)
        if use_pid:
            actual_angles_rad = self.update_pid(target_angles_rad)
        else:
            actual_angles_rad = target_angles_rad
            self.current_positions_rad = actual_angles_rad

        # Step 3: Convert to Dynamixel positions
        dynamixel_positions = self.radians_to_dynamixel(actual_angles_rad)

        if verbose:
            print(f"\nJoint Mapping (with PID smoothing):")
            for i, name in enumerate(self.joint_names):
                target_deg = np.rad2deg(target_angles_rad[i])
                actual_deg = np.rad2deg(actual_angles_rad[i])
                error_deg = target_deg - actual_deg
                print(f"  {name:8s}: action={policy_actions[i]:+.2f} → "
                      f"target={target_deg:+7.2f}° → "
                      f"actual={actual_deg:+7.2f}° (err={error_deg:+.2f}°) → "
                      f"position={dynamixel_positions[i]:4d}")

        return dynamixel_positions

    def reset(self, initial_dynamixel_positions=None):
        """
        Reset PID state (call at start of new episode).

        Args:
            initial_dynamixel_positions: Optional 8D array of starting Dynamixel positions.
                                        If provided, initializes PID from these positions.
                                        If None, uses MuJoCo centers (flat position).
        """
        if initial_dynamixel_positions is not None:
            # Convert Dynamixel positions back to radians for PID state
            # This allows PID to start from standing position!
            # Inverse of radians_to_dynamixel mapping

            # Normalize Dynamixel position to [0, 1] within each joint's range
            normalized = (initial_dynamixel_positions - self.dynamixel_ranges[:, 0]) / self.dynamixel_spans

            # Map to MuJoCo range in degrees
            initial_degrees = self.mujoco_ranges[:, 0] + normalized * self.mujoco_spans

            # Convert to radians
            self.current_positions_rad = np.deg2rad(initial_degrees)
            self.target_positions_rad = self.current_positions_rad.copy()
        else:
            # Default: start from MuJoCo centers (flat position)
            self.current_positions_rad = np.deg2rad(self.mujoco_centers)
            self.target_positions_rad = np.deg2rad(self.mujoco_centers)

        self.past_errors = np.zeros(8)


# Test the calibration with PID
if __name__ == "__main__":
    print("="*70)
    print("RealANT Joint Calibration with PID Test")
    print("="*70)

    calibration = RealAntJointCalibrationPID(dt=0.02, verbose=True)

    print("\n" + "="*70)
    print("Test: Binary action sequence (like real policy!)")
    print("="*70)

    # Simulate binary bang-bang control like the actual policy
    test_sequence = [
        np.array([-1, -1, 1, 1, 1, -1, -1, 1]),  # Step 1
        np.array([-1, -1, 1, 1, 1, -1, -1, 1]),  # Step 2 (hold)
        np.array([1, 1, -1, -1, -1, 1, 1, -1]),  # Step 3 (flip)
        np.array([1, 1, -1, -1, -1, 1, 1, -1]),  # Step 4 (hold)
    ]

    print("\nWithout PID (direct mapping - causes violent jumps):")
    calibration.reset()
    for step, actions in enumerate(test_sequence):
        positions = calibration.policy_action_to_dynamixel(actions, use_pid=False, verbose=False)
        print(f"  Step {step+1}: {positions}")

    print("\nWith PID (smooth like MuJoCo - coordinated motion):")
    calibration.reset()
    for step, actions in enumerate(test_sequence):
        positions = calibration.policy_action_to_dynamixel(actions, use_pid=True, verbose=False)
        print(f"  Step {step+1}: {positions}")

    print("\n✅ Notice how PID smooths the binary jumps into gradual movement!")
