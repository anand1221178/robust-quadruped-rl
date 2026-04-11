#!/usr/bin/env python3
"""
Direct Policy Deployment to RealANT Robot

This script deploys YOUR trained robustness policies (M1-M4) directly to the
physical RealANT robot using the native ZeroMQ architecture.

NO ROS REQUIRED! This uses the original RealANT design:
- ZeroMQ for inter-process communication (lightweight, real-time)
- AR tag tracking for pose estimation (via OpenCV ArUco)
- Direct Dynamixel servo control (via USB serial)

Usage:
    # Terminal 1: Start AR tag tracking
    python ar_tag_tracker.py --camera 0 --port 5555

    # Terminal 2: Start servo controller
    python servo_controller.py --port /dev/ttyUSB0 --sub-port 5556 --pub-port 5557

    # Terminal 3: Deploy YOUR trained policy
    python deploy_robust_policy.py \\
        --model "../../done/v7_7e_ultra_speed_jtfwl2qf/best_model/best_model.zip" \\
        --vec-normalize "../../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl" \\
        --episodes 10

Author: Robust Quadruped RL Research Team
Based on: https://github.com/AaltoVision/realant-rl
"""

import zmq
import numpy as np
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


class RobustPolicyDeployer:
    """
    Deploys trained PPO policies to physical RealANT robot.

    This class:
    1. Loads YOUR trained model + VecNormalize stats
    2. Connects to RealANT hardware via ZeroMQ (NO ROS!)
    3. Runs real-time control loop at 20Hz
    4. Handles observation stacking for latency (like in training)
    """

    def __init__(
        self,
        model_path: str,
        vec_normalize_path: str,
        obs_port: int = 5555,
        joint_port: int = 5557,
        action_port: int = 5556,
        control_freq: float = 20.0,
        obs_stack_size: int = 4
    ):
        """
        Initialize deployment system.

        Args:
            model_path: Path to trained model (.zip file)
            vec_normalize_path: Path to VecNormalize stats (.pkl file)
            obs_port: ZMQ port for AR tag pose observations
            joint_port: ZMQ port for servo joint states
            action_port: ZMQ port for sending servo commands
            control_freq: Control loop frequency (Hz)
            obs_stack_size: Number of observations to stack (handles latency)
        """
        self.control_freq = control_freq
        self.control_period = 1.0 / control_freq
        self.obs_stack_size = obs_stack_size

        print(f"\n{'='*60}")
        print(f"🤖 RealANT Robust Policy Deployment System")
        print(f"{'='*60}\n")

        # 1. Load trained model
        print(f"Loading trained model...")
        print(f"   Path: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = PPO.load(model_path)
        print(f"   Model loaded successfully!")

        # 2. Load VecNormalize statistics (CRITICAL!)
        print(f"\nLoading normalization statistics...")
        print(f"   Path: {vec_normalize_path}")
        if not Path(vec_normalize_path).exists():
            raise FileNotFoundError(f"VecNormalize not found: {vec_normalize_path}")

        self.vec_normalize = VecNormalize.load(
            vec_normalize_path,
            DummyVecEnv([lambda: None])  # Dummy env for loading
        )
        print(f"   Normalization stats loaded!")
        print(f"   Obs mean: {self.vec_normalize.obs_rms.mean[:5]}...")
        print(f"   Obs std:  {np.sqrt(self.vec_normalize.obs_rms.var[:5])}...")

        # 3. Setup ZeroMQ communication
        print(f"\nSetting up ZeroMQ communication...")
        context = zmq.Context()

        # Subscribe to AR tag pose observations
        print(f"   Connecting to AR tag tracker (port {obs_port})...")
        self.obs_socket = context.socket(zmq.SUB)
        self.obs_socket.connect(f"tcp://localhost:{obs_port}")
        self.obs_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.obs_socket.setsockopt(zmq.CONFLATE, 1)  # Only keep latest message

        # Subscribe to joint state observations
        print(f"   Connecting to servo controller (port {joint_port})...")
        self.joint_socket = context.socket(zmq.SUB)
        self.joint_socket.connect(f"tcp://localhost:{joint_port}")
        self.joint_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.joint_socket.setsockopt(zmq.CONFLATE, 1)

        # Publish actions to servos
        print(f"   Binding action publisher (port {action_port})...")
        self.action_socket = context.socket(zmq.PUB)
        self.action_socket.bind(f"tcp://*:{action_port}")
        time.sleep(0.5)  # Wait for socket to bind

        print(f"   ZeroMQ setup complete!")

        # 4. Initialize state
        self.obs_history = []
        self.last_pose = None
        self.last_joint_state = None

        print(f"\nInitialization complete!\n")

    def get_observation(self, timeout_ms: int = 50) -> Optional[np.ndarray]:
        """
        Get current 29D observation from robot sensors.

        Constructs observation from:
        - AR tag tracking (pose + velocities)
        - Servo encoders (joint positions + velocities)

        Returns:
            29D numpy array or None if data not available
        """
        # Receive AR tag pose (non-blocking with timeout)
        try:
            if self.obs_socket.poll(timeout_ms):
                pose_msg = self.obs_socket.recv_json(flags=zmq.NOBLOCK)
                self.last_pose = pose_msg
        except zmq.Again:
            pass

        # Receive joint states (non-blocking with timeout)
        try:
            if self.joint_socket.poll(timeout_ms):
                joint_msg = self.joint_socket.recv_json(flags=zmq.NOBLOCK)
                self.last_joint_state = joint_msg
        except zmq.Again:
            pass

        # Check if we have both pose and joint data
        if self.last_pose is None or self.last_joint_state is None:
            return None

        # Construct 29D observation (EXACTLY matching simulation)
        try:
            obs = np.array([
                # Torso velocities (3D)
                self.last_pose['vx'],
                self.last_pose['vy'],
                self.last_pose['vz'],

                # Torso z position (1D)
                self.last_pose['z'],

                # Torso orientation as sin/cos of Euler angles (6D)
                np.sin(self.last_pose['roll']),
                np.cos(self.last_pose['roll']),
                np.sin(self.last_pose['pitch']),
                np.cos(self.last_pose['pitch']),
                np.sin(self.last_pose['yaw']),
                np.cos(self.last_pose['yaw']),

                # Torso angular velocities (3D)
                self.last_pose['vroll'],
                self.last_pose['vpitch'],
                self.last_pose['vyaw'],

                # Joint positions (8D)
                *self.last_joint_state['positions'],

                # Joint velocities (8D)
                *self.last_joint_state['velocities'],
            ], dtype=np.float32)

            return obs
        except (KeyError, TypeError, ValueError) as e:
            print(f"⚠️  Error constructing observation: {e}")
            return None

    def send_action(self, action: np.ndarray):
        """
        Send action to Dynamixel servos.

        Converts policy action [-1, 1] to Dynamixel setpoints [224, 800].

        Args:
            action: 8D action from policy network
        """
        # Convert to Dynamixel range
        servo_commands = self.action_to_dynamixel(action)

        # Format as Dynamixel command string
        cmd_parts = [f"s{i+1} {int(cmd)}" for i, cmd in enumerate(servo_commands)]
        action_msg = {
            "cmd": " ".join(cmd_parts),
            "timestamp": time.time()
        }

        # Send via ZMQ
        self.action_socket.send_json(action_msg)

    def action_to_dynamixel(self, action: np.ndarray) -> np.ndarray:
        """
        Convert policy action to Dynamixel servo setpoints.

        Policy outputs: [-1, 1] (normalized joint angles)
        Dynamixel range: [0, 1023] (raw servo positions)
        Safe range: [224, 800] (prevents mechanical damage)

        Args:
            action: 8D normalized action from policy

        Returns:
            8D Dynamixel setpoints
        """
        # Center: 512, max deviation: ±288
        setpoints = 512 + action * 288

        # Clip to safe range
        setpoints = np.clip(setpoints, 224, 800)

        return setpoints

    def run_episode(
        self,
        episode_length: int = 200,
        warmup_steps: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Run one episode on the real robot.

        Args:
            episode_length: Number of steps (default 200 = 10s @ 20Hz)
            warmup_steps: Initial steps to fill observation history
            verbose: Print progress updates

        Returns:
            Dictionary with episode statistics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Starting Episode")
            print(f"{'='*60}")
            print(f"   Duration: {episode_length} steps ({episode_length * self.control_period:.1f}s)")
            print(f"   Control Frequency: {self.control_freq}Hz")
            print(f"   Warmup Steps: {warmup_steps}")
            print(f"\n⚠️  MAKE SURE ROBOT IS IN STARTING POSITION!\n")

        # Reset state
        self.obs_history = []
        episode_reward = 0.0
        episode_distance = 0.0
        start_position = None
        steps_completed = 0
        failed_obs_count = 0

        # Warmup: fill observation history
        if verbose:
            print(f"🔄 Warming up observation buffer...")

        for _ in range(warmup_steps):
            obs = self.get_observation(timeout_ms=100)
            if obs is not None:
                self.obs_history.append(obs)
                if start_position is None and len(obs) > 0:
                    start_position = obs[0]  # x velocity

        if len(self.obs_history) == 0:
            print(f"❌ ERROR: No observations received during warmup!")
            print(f"   Make sure AR tag tracker and servo controller are running.")
            return {"success": False, "error": "no_observations"}

        if verbose:
            print(f"   ✅ Buffer filled with {len(self.obs_history)} observations\n")
            print(f"{'─'*60}")
            print(f"{'Step':>4} | {'X-Vel':>7} | {'Z-Pos':>7} | {'Reward':>8} | {'Freq':>6}")
            print(f"{'─'*60}")

        # Main control loop
        for step in range(episode_length):
            step_start = time.time()

            # 1. Get observation from robot
            obs = self.get_observation(timeout_ms=50)

            if obs is None:
                failed_obs_count += 1
                if failed_obs_count > 20:
                    print(f"\n❌ ERROR: Too many failed observations ({failed_obs_count})")
                    break
                time.sleep(self.control_period)
                continue

            failed_obs_count = 0
            steps_completed += 1

            # 2. Stack observations (handles latency)
            self.obs_history.append(obs)
            if len(self.obs_history) > self.obs_stack_size:
                self.obs_history.pop(0)

            # 3. Normalize observation (using training stats)
            obs_normalized = self.vec_normalize.normalize_obs(obs)

            # 4. Get action from trained policy
            action, _ = self.model.predict(obs_normalized, deterministic=True)

            # 5. Send action to servos
            self.send_action(action)

            # 6. Compute reward (for monitoring)
            reward = obs[0]  # Forward velocity (x-axis)
            episode_reward += reward
            episode_distance += abs(obs[0]) * self.control_period

            # 7. Log progress
            if verbose and (step % 20 == 0 or step < 10):
                actual_freq = 1.0 / (time.time() - step_start) if step > 0 else 0
                print(f"{step:4d} | {obs[0]:7.3f} | {obs[3]:7.3f} | {episode_reward:8.2f} | {actual_freq:6.1f}Hz")

            # 8. Maintain control frequency
            elapsed = time.time() - step_start
            sleep_time = max(0, self.control_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Episode complete
        if verbose:
            print(f"{'─'*60}")
            print(f"\n✅ Episode Complete!")
            print(f"   Steps: {steps_completed}/{episode_length}")
            print(f"   Total Reward: {episode_reward:.2f}")
            print(f"   Distance: {episode_distance:.2f}m")
            print(f"   Failed Observations: {failed_obs_count}")
            print(f"{'='*60}\n")

        return {
            "success": True,
            "steps": steps_completed,
            "reward": episode_reward,
            "distance": episode_distance,
            "failed_obs": failed_obs_count
        }

    def test_communication(self, duration: float = 5.0) -> bool:
        """
        Test ZeroMQ communication with robot hardware.

        Args:
            duration: Test duration in seconds

        Returns:
            True if communication working
        """
        print(f"\n🔍 Testing Communication ({duration}s)...")
        print(f"{'─'*60}")

        start_time = time.time()
        pose_count = 0
        joint_count = 0

        while time.time() - start_time < duration:
            obs = self.get_observation(timeout_ms=100)
            if obs is not None:
                if self.last_pose is not None:
                    pose_count += 1
                if self.last_joint_state is not None:
                    joint_count += 1
            time.sleep(0.1)

        print(f"\n📊 Communication Test Results:")
        print(f"   AR Tag Messages: {pose_count}")
        print(f"   Joint State Messages: {joint_count}")

        if pose_count > 0 and joint_count > 0:
            print(f"   ✅ Communication working!")
            return True
        else:
            print(f"   ❌ Communication failed!")
            if pose_count == 0:
                print(f"      - No AR tag data (is ar_tag_tracker.py running?)")
            if joint_count == 0:
                print(f"      - No joint data (is servo_controller.py running?)")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Deploy trained robustness policies to RealANT robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy M3 (DR champion) - RECOMMENDED for real robot
  python deploy_robust_policy.py \\
      --model ../../done/v7_7e_ultra_speed_jtfwl2qf/best_model/best_model.zip \\
      --vec-normalize ../../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl

  # Deploy M2 (SR2L) with 5 episodes
  python deploy_robust_policy.py \\
      --model ../../done/ppo_sr2l_forward_m7gtjtpa/final_model.zip \\
      --vec-normalize ../../done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl \\
      --episodes 5

  # Test communication only
  python deploy_robust_policy.py \\
      --model ../../done/v7_7e_ultra_speed_jtfwl2qf/best_model/best_model.zip \\
      --vec-normalize ../../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl \\
      --test-only
        """
    )

    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--vec-normalize", required=True, help="Path to VecNormalize stats (.pkl)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--episode-length", type=int, default=200, help="Steps per episode (200=10s)")
    parser.add_argument("--control-freq", type=float, default=20.0, help="Control frequency (Hz)")
    parser.add_argument("--test-only", action="store_true", help="Only test communication, don't run episodes")
    parser.add_argument("--obs-port", type=int, default=5555, help="AR tag observation port")
    parser.add_argument("--joint-port", type=int, default=5557, help="Joint state port")
    parser.add_argument("--action-port", type=int, default=5556, help="Action command port")

    args = parser.parse_args()

    try:
        # Initialize deployer
        deployer = RobustPolicyDeployer(
            model_path=args.model,
            vec_normalize_path=args.vec_normalize,
            obs_port=args.obs_port,
            joint_port=args.joint_port,
            action_port=args.action_port,
            control_freq=args.control_freq
        )

        # Test communication
        if not deployer.test_communication(duration=3.0):
            print(f"\n❌ Communication test failed!")
            print(f"\n💡 Make sure these processes are running:")
            print(f"   1. ar_tag_tracker.py (AR tag tracking)")
            print(f"   2. servo_controller.py (servo control)")
            return 1

        if args.test_only:
            print(f"\n✅ Test complete! Communication working.")
            return 0

        # Run episodes
        print(f"\n🎯 Starting Deployment")
        print(f"   Model: {Path(args.model).name}")
        print(f"   Episodes: {args.episodes}")
        print(f"   Episode Length: {args.episode_length} steps ({args.episode_length/args.control_freq:.1f}s)")
        print(f"\n⏳ Starting in 3 seconds... (Ctrl+C to cancel)")
        time.sleep(3)

        results = []
        for ep in range(args.episodes):
            print(f"\n{'#'*60}")
            print(f"# Episode {ep+1}/{args.episodes}")
            print(f"{'#'*60}")

            result = deployer.run_episode(episode_length=args.episode_length)

            if result["success"]:
                results.append(result)
                print(f"\n📊 Episode {ep+1} Summary:")
                print(f"   Reward: {result['reward']:.2f}")
                print(f"   Distance: {result['distance']:.2f}m")
            else:
                print(f"\n❌ Episode {ep+1} failed!")
                break

            if ep < args.episodes - 1:
                print(f"\n⏸️  Manually reset robot position, then press Enter...")
                input()

        # Final summary
        if results:
            print(f"\n{'='*60}")
            print(f"🏁 Deployment Complete!")
            print(f"{'='*60}")
            rewards = [r["reward"] for r in results]
            distances = [r["distance"] for r in results]
            print(f"   Episodes: {len(results)}")
            print(f"   Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"   Avg Distance: {np.mean(distances):.2f} ± {np.std(distances):.2f}m")
            print(f"   Best Distance: {np.max(distances):.2f}m")
            print(f"{'='*60}\n")

        return 0

    except KeyboardInterrupt:
        print(f"\n\n⏹️  Deployment interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
