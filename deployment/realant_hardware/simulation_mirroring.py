#!/usr/bin/env python3
"""
Simulation-to-Robot Mirroring for RealANT

This script runs YOUR trained policy in MuJoCo simulation and mirrors the
simulated joint positions to the physical robot in real-time.

This approach is EASIER for initial testing because:
- You can SEE what the policy is doing (MuJoCo visualization)
- Easier to debug sim-to-real gaps (compare simulated vs actual behavior)
- Can add safety limits before sending to hardware
- Can record both simulation and robot simultaneously

Architecture:
    ┌─────────────────────────────┐
    │  MuJoCo Simulation          │
    │  - Loads M3 policy          │
    │  - Runs at 50Hz             │
    │  - Shows visualization      │
    │  - Policy sees sim obs      │
    └──────────┬──────────────────┘
               │ Extract joint positions
               │ every timestep
               ▼
    ┌─────────────────────────────┐
    │  ZeroMQ Bridge              │
    │  - Converts to Dynamixel    │
    │  - Applies safety limits    │
    │  - Sends @ 20Hz             │
    └──────────┬──────────────────┘
               │ "cmd": "s1 X s2 Y..."
               ▼
    ┌─────────────────────────────┐
    │  Physical RealANT Robot     │
    │  - Servos mirror simulation │
    │  - Optional: AR tag logging │
    └─────────────────────────────┘

Usage:
    # Terminal 1: Start servo controller
    cd deployment/realant_hardware/realant
    python ant_server.py --port /dev/ttyUSB0

    # Terminal 2: Run simulation mirroring
    cd deployment/realant_hardware
    python simulation_mirroring.py \\
        --model ../../experiments/M3_dr_seed42/best_model/best_model.zip \\
        --vec-normalize ../../experiments/M3_dr_seed42/vec_normalize.pkl \\
        --episodes 5 \\
        --render  # Shows MuJoCo visualization

Author: Robust Quadruped RL Research Team
"""

import zmq
import numpy as np
import time
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym

# Import your custom RealAnt environment
sys.path.insert(0, str(project_root / "src"))
import realant_sim  # This registers the environment with gymnasium

# Import PID-based joint calibration (CRITICAL for binary actions!)
from joint_calibration_pid import RealAntJointCalibrationPID


class SimulationMirroringBridge:
    """
    Bridge between MuJoCo simulation and physical RealANT robot.

    This class:
    1. Runs your trained policy in MuJoCo simulation (with visualization!)
    2. Extracts joint positions from simulation every step
    3. Sends joint positions to physical robot via ZeroMQ
    4. Applies safety limits to prevent hardware damage
    """

    def __init__(
        self,
        model_path: str,
        vec_normalize_path: str,
        render: bool = True,
        mirror_freq: float = 20.0,
        sim_freq: float = 50.0,
        action_port: int = 3001,
        safety_limits: bool = True,
        action_smoothing: float = 1.0,
        action_scale: float = 1.0,
        rate_limit: int = 50,
        use_pid: bool = True
    ):
        """
        Initialize simulation mirroring system.

        Args:
            model_path: Path to trained PPO model (.zip)
            vec_normalize_path: Path to VecNormalize stats (.pkl)
            render: Show MuJoCo visualization
            mirror_freq: Frequency to send commands to robot (Hz)
            sim_freq: Simulation frequency (Hz)
            action_port: ZeroMQ port for sending servo commands
            safety_limits: Apply safety checks before sending to hardware
            action_smoothing: Alpha for EMA filter (0-1). 1=off.
            action_scale: Scale factor for policy actions (0.5 = half speed).
            rate_limit: Max change in servo position per step (0 to disable).
            use_pid: Whether to use the software PID controller.
        """
        self.render = render
        self.mirror_freq = mirror_freq
        self.sim_freq = sim_freq
        self.mirror_period = 1.0 / mirror_freq
        self.safety_limits = safety_limits
        self.action_smoothing_alpha = action_smoothing
        self.action_scale = action_scale
        self.rate_limit = rate_limit
        self.use_pid = use_pid
        self.filtered_action = np.zeros(8)
        self.last_servo_commands = None

        print(f"\n{'='*70}")
        print(f"🎮 RealANT Simulation-to-Robot Mirroring System")
        print(f"{'='*70}\n")

        # 1. Create MuJoCo simulation environment
        print(f"Creating MuJoCo simulation environment...")

        def make_env():
            env = gym.make(
                'RealAntMujoco-v0',
                render_mode='human' if render else None,
                max_episode_steps=1000
            )
            return env

        self.vec_env = DummyVecEnv([make_env])
        print(f"   Environment created!")

        # 2. Load VecNormalize statistics
        print(f"\nLoading normalization statistics...")
        print(f"   Path: {vec_normalize_path}")

        if not Path(vec_normalize_path).exists():
            raise FileNotFoundError(f"VecNormalize not found: {vec_normalize_path}")

        self.vec_env = VecNormalize.load(vec_normalize_path, self.vec_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        print(f"   Normalization loaded!")

        # 3. Load trained policy
        print(f"\nLoading trained policy...")
        print(f"   Path: {model_path}")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = PPO.load(model_path)
        print(f"   Model loaded successfully!")

        # 4. Setup ZeroMQ for robot communication
        print(f"\nSetting up ZeroMQ communication...")
        context = zmq.Context()

        # Command socket (send to robot)
        self.action_socket = context.socket(zmq.PUB)
        self.action_socket.setsockopt(zmq.SNDHWM, 10)  # Limit send buffer to 10 messages
        self.action_socket.bind(f"tcp://*:{action_port}")
        print(f"   Bound to port {action_port} (send commands)")

        # Sensor socket (receive from robot) - PARTIAL CLOSED-LOOP!
        self.sensor_socket = context.socket(zmq.SUB)
        self.sensor_socket.connect("tcp://127.0.0.1:3002")
        self.sensor_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sensor_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        print(f"   Connected to port 3002 (receive sensor data)")
        print(f"   🔄 Partial closed-loop: Using REAL joint positions!")

        time.sleep(0.5)  # Wait for connections

        # Tracking for real joint feedback
        self.last_real_joint_positions = np.zeros(8)  # in Dynamixel units
        self.last_real_joint_radians = np.zeros(8)  # converted to radians

        # 5. Initialize PID-based joint calibration (CRITICAL for binary actions!)
        print(f"\n🎯 Loading joint calibration with PID smoothing...")
        self.joint_calibration = RealAntJointCalibrationPID(dt=1.0/self.sim_freq, verbose=False)
        print(f"   ✅ PID calibration loaded!")
        print(f"   ℹ️  Policy outputs BINARY actions - PID smooths them like MuJoCo!")

        # 6. Initialize state tracking
        self.last_mirror_time = 0.0
        self.mirror_count = 0
        self.sim_step_count = 0

        print(f"\n✅ Initialization complete!")
        print(f"\n{'─'*70}")
        print(f"Configuration:")
        print(f"   Simulation Frequency: {self.sim_freq}Hz")
        print(f"   Robot Mirror Frequency: {self.mirror_freq}Hz")
        print(f"   Render Visualization: {self.render}")
        print(f"   Safety Limits: {self.safety_limits}")
        if self.action_smoothing_alpha < 1.0:
            print(f"   Action Smoothing: ENABLED (alpha={self.action_smoothing_alpha})")
        else:
            print(f"   Action Smoothing: DISABLED")
        
        print(f"   Action Scale: {self.action_scale}")
        print(f"   Rate Limiting: {self.rate_limit} pos/step")
        print(f"   Software PID: {'ENABLED' if self.use_pid else 'DISABLED'}")

        print(f"{'─'*70}\n")

    def read_real_joint_positions(self):
        """
        Read actual joint positions from robot servos (PARTIAL CLOSED-LOOP!).

        Returns:
            Joint positions in radians (in MuJoCo coordinate system!)
        """
        try:
            # Try to read sensor data (non-blocking with timeout)
            msg = self.sensor_socket.recv_json()

            # Parse sensor data (from ant_server format)
            if 's1_angle' in msg:
                # Extract actual positions (s1_angle through s8_angle)
                positions_dynamixel = np.array([
                    float(msg.get(f's{i+1}_angle', 512))
                    for i in range(8)
                ])

                # CRITICAL: Convert Dynamixel positions to MuJoCo radians
                # Each joint has a different center! Use calibration info.

                # Offset from Dynamixel center (512)
                offsets_dynamixel = positions_dynamixel - 512

                # Convert to degrees offset
                offsets_degrees = offsets_dynamixel / 3.41  # ~3.41 positions per degree

                # Add to each joint's MuJoCo center angle
                # Centers: [0, 65, 0, -65, 0, -65, 0, 65] degrees
                joint_centers_deg = self.joint_calibration.mujoco_centers
                positions_degrees = joint_centers_deg + offsets_degrees

                # Convert to radians for MuJoCo
                positions_radians = np.deg2rad(positions_degrees)

                # Store for velocity estimation
                self.last_real_joint_positions = positions_dynamixel
                self.last_real_joint_radians = positions_radians

                return positions_radians

        except zmq.Again:
            # Timeout - no new data, use last known
            return self.last_real_joint_radians
        except Exception as e:
            # Parse error - use last known
            return self.last_real_joint_radians

    def update_observation_with_real_joints(self, sim_obs: np.ndarray) -> np.ndarray:
        """
        Replace simulated joint positions with REAL servo positions.

        Observation structure (29D):
        - [0:3]   body_xyz_vel (use sim)
        - [3:4]   body_z (use sim)
        - [4:7]   body_rpy_vel (use sim)
        - [7:10]  sin(body_rpy) (use sim)
        - [10:13] cos(body_rpy) (use sim)
        - [13:21] joint_positions (REPLACE with real!)
        - [21:29] joint_velocities (REPLACE with real!)

        Args:
            sim_obs: Observation from vectorized env, shape (1, 29)

        Returns:
            Observation with batch dim, shape (1, 29) - hybrid (sim body + real joints)
        """
        # Store the previous joint positions before reading new ones
        prev_joint_pos_rad = self.last_real_joint_radians.copy()

        # Read real joint positions (this updates self.last_real_joint_radians)
        real_joint_pos = self.read_real_joint_positions()

        # Estimate velocities using finite difference
        dt = 1.0 / self.sim_freq
        real_joint_vel = (real_joint_pos - prev_joint_pos_rad) / dt

        # Build hybrid observation (handle batch dimension from VecEnv)
        hybrid_obs = sim_obs.copy()

        # VecEnv returns shape (1, 29), so index into batch dimension
        if hybrid_obs.ndim == 2:
            hybrid_obs[0, 13:21] = real_joint_pos  # Real positions!
            hybrid_obs[0, 21:29] = real_joint_vel  # REAL velocities!
        else:
            # Fallback for 1D case
            hybrid_obs[13:21] = real_joint_pos
            hybrid_obs[21:29] = real_joint_vel

        return hybrid_obs

    def reset_to_neutral_position(self, verbose: bool = True):
        """
        Move robot to neutral standing position at the start of each episode.

        This prevents the robot from starting in whatever weird position it ended
        the last episode in.
        """
        if verbose:
            print(f"\n🤖 Moving robot to neutral position...")

        # Standing position = hips neutral, ankles fully extended (legs at 90°)
        # Direct Dynamixel positions (not using calibration for initial pose)
        # Hips at 512 (neutral), ankles at 224 (fully extended standing)
        neutral_positions = np.array([
            512,  # s1 hip_1: neutral
            224,  # s2 ankle_1: standing (fully extended)
            512,  # s3 hip_2: neutral
            224,  # s4 ankle_2: standing (fully extended)
            512,  # s5 hip_3: neutral
            224,  # s6 ankle_3: standing (fully extended)
            512,  # s7 hip_4: neutral
            224   # s8 ankle_4: standing (fully extended)
        ])

        if verbose:
            print(f"   Target positions: {neutral_positions}")

        # Send neutral position command
        cmd_parts = [f"s{i+1} {int(pos)}" for i, pos in enumerate(neutral_positions)]
        cmd_string = " ".join(cmd_parts) + "\n"

        try:
            self.action_socket.send_multipart([b"cmd", cmd_string.encode()], flags=zmq.NOBLOCK)
            if verbose:
                print(f"   Waiting 3 seconds for servos to reach position...")
            time.sleep(3.0)  # Give servos time to move
            if verbose:
                print(f"   ✅ Robot in standing position!")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not send neutral position: {e}")

        # Initialize rate limiter state
        self.last_servo_commands = neutral_positions.copy()

        return neutral_positions  # Return for PID initialization

    def send_joint_positions_to_robot(self, joint_positions: np.ndarray, debug: bool = False):
        """
        Send joint positions from simulation to physical robot.

        Args:
            joint_positions: 8D array of joint positions from simulation
            debug: Print detailed mapping info
        """
        # Convert joint positions to Dynamixel servo commands
        servo_commands = self.joint_positions_to_dynamixel(joint_positions)

        # Debug output every 100 commands
        if debug and self.mirror_count % 100 == 0:
            print(f"\n{'='*70}")
            print(f"Debug - Command #{self.mirror_count}")
            print(f"Policy actions:  {joint_positions}")
            print(f"Servo positions: {servo_commands}")
            print(f"  hip_1 (s1): {servo_commands[0]:4d}  |  ankle_1 (s2): {servo_commands[1]:4d}")
            print(f"  hip_2 (s3): {servo_commands[2]:4d}  |  ankle_2 (s4): {servo_commands[3]:4d}")
            print(f"  hip_3 (s5): {servo_commands[4]:4d}  |  ankle_3 (s6): {servo_commands[5]:4d}")
            print(f"  hip_4 (s7): {servo_commands[6]:4d}  |  ankle_4 (s8): {servo_commands[7]:4d}")
            print(f"{'='*70}\n")

        # Apply safety limits if enabled
        if self.safety_limits:
            servo_commands = self.apply_safety_limits(servo_commands)

        # Format as Dynamixel command string (must start with "cmd" topic!)
        cmd_parts = [f"s{i+1} {int(cmd)}" for i, cmd in enumerate(servo_commands)]
        cmd_string = " ".join(cmd_parts) + "\n"  # CRITICAL: Add newline!

        # Send via ZeroMQ with "cmd" topic (required by ant_server!)
        # NO NOBLOCK flag - we want to wait and send every command (no skipping!)
        # Skipping commands causes violent jumps between positions
        try:
            self.action_socket.send_multipart([b"cmd", cmd_string.encode()])
            self.mirror_count += 1
        except Exception as e:
            print(f"⚠️  Warning: Failed to send command: {e}")

    def joint_positions_to_dynamixel(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Convert policy actions to Dynamixel servo setpoints using PROPER calibration!

        This uses the RealAntJointCalibration class to correctly map each joint
        based on its specific range and center from the MuJoCo XML.

        Args:
            joint_positions: 8D policy actions in [-1, 1] range

        Returns:
            8D Dynamixel setpoints [224-800]
        """
        # 1. Action scaling (user-defined)
        # 2. EMA smoothing (user-defined)
        # 3. Rate limiting (user-defined)
        setpoints = self.joint_calibration.policy_action_to_dynamixel(
            joint_positions,
            use_pid=self.use_pid
        )

        return setpoints

    def apply_safety_limits(self, servo_commands: np.ndarray) -> np.ndarray:
        """
        Apply safety limits to prevent hardware damage.

        Args:
            servo_commands: 8D Dynamixel setpoints

        Returns:
            8D safe Dynamixel setpoints
        """
        # Apply rate limiting to prevent violent jumps
        if self.rate_limit > 0 and self.last_servo_commands is not None:
            delta = servo_commands - self.last_servo_commands
            delta = np.clip(delta, -self.rate_limit, self.rate_limit)
            servo_commands = self.last_servo_commands + delta

        # Hard limits: [224, 800]
        servo_commands = np.clip(servo_commands, 224, 800)

        # Store the final command for the next iteration's rate limiting
        self.last_servo_commands = servo_commands.copy()

        return servo_commands

    def run_episode(
        self,
        episode_length: int = 1000,
        verbose: bool = True
    ) -> Dict:
        """
        Run one episode with simulation mirroring.

        Args:
            episode_length: Max steps per episode
            verbose: Print progress

        Returns:
            Episode statistics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🚀 Starting Mirroring Episode")
            print(f"{'='*70}")
            print(f"   Episode Length: {episode_length} steps")
            print(f"   Duration: ~{episode_length / self.sim_freq:.1f}s")
            print(f"\n⚠️  MAKE SURE:")
            print(f"   1. Servo controller (ant_server.py) is running")
            print(f"   2. Robot is in starting position")
            print(f"   3. Workspace is clear of obstacles\n")
            input("Press Enter to start episode...")

        # Attach servos before starting
        if verbose:
            print(f"\n📌 Attaching servos...")
        try:
            self.action_socket.send_multipart([b"cmd", b"attach_servos\n"])
            time.sleep(0.2)
        except:
            pass

        # Move robot to standing position first!
        standing_positions = self.reset_to_neutral_position(verbose=verbose)

        # CRITICAL: Reset PID controller state to match standing position!
        # This prevents robot from collapsing back to flat
        self.joint_calibration.reset(initial_dynamixel_positions=standing_positions)
        if verbose:
            print(f"   🔄 PID controller initialized from standing position")

        # Reset simulation
        obs = self.vec_env.reset()

        # CRITICAL: Give robot 50 steps to stabilize/stand up before mirroring!
        if verbose:
            print(f"🔧 Warming up simulation (50 steps)...")
        for warmup_step in range(50):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, _, _ = self.vec_env.step(action)
            # Don't mirror during warmup!

        if verbose:
            print(f"✅ Robot should be standing now!")

        # Reset EMA filter at start of episode
        self.filtered_action = np.zeros(8)

        episode_reward = 0.0
        episode_distance = 0.0
        step = 0
        done = False

        self.last_mirror_time = time.time()
        episode_start_time = time.time()

        if verbose:
            print(f"\n{'─'*70}")
            print(f"{'Step':>6} | {'Reward':>8} | {'Distance':>9} | {'SimHz':>6} | {'MirrorHz':>9}")
            print(f"{'─'*70}")

        # Main simulation loop with PID SMOOTHING!
        while step < episode_length and not done:
            step_start = time.time()

            # 1. Get action from policy
            action, _ = self.model.predict(obs, deterministic=True)

            # 2. Smooth action and send to robot
            raw_joint_positions = action[0]

            # Scale down actions to prevent extreme movements
            scaled_actions = raw_joint_positions * self.action_scale

            # Apply Exponential Moving Average (EMA) to smooth actions
            self.filtered_action = (
                self.action_smoothing_alpha * scaled_actions +
                (1 - self.action_smoothing_alpha) * self.filtered_action
            )

            current_time = time.time()
            if current_time - self.last_mirror_time >= self.mirror_period:
                self.send_joint_positions_to_robot(self.filtered_action, debug=True)
                self.last_mirror_time = current_time

            # 3. Step simulation
            obs, reward, done, info = self.vec_env.step(action)
            episode_reward += reward[0]

            # 5. Track distance
            if len(info[0]) > 0 and 'x_position' in info[0]:
                episode_distance = info[0]['x_position']

            # 6. Logging
            if verbose and (step % 50 == 0 or step < 10):
                elapsed = time.time() - episode_start_time
                sim_hz = step / elapsed if elapsed > 0 else 0
                mirror_hz = self.mirror_count / elapsed if elapsed > 0 else 0
                print(f"{step:6d} | {episode_reward:8.2f} | {episode_distance:9.3f}m | {sim_hz:6.1f} | {mirror_hz:9.1f}")

            # 7. Maintain simulation frequency (CRITICAL for real-time sync)
            step += 1
            self.sim_step_count += 1

            # Always sleep to maintain real-time sync (even with rendering)
            elapsed = time.time() - step_start
            sleep_time = max(0, 1.0/self.sim_freq - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Episode complete
        total_time = time.time() - episode_start_time

        if verbose:
            print(f"{'─'*70}")
            print(f"\n✅ Episode Complete!")
            print(f"   Total Steps: {step}")
            print(f"   Duration: {total_time:.1f}s")
            print(f"   Total Reward: {episode_reward:.2f}")
            print(f"   Final Distance: {episode_distance:.3f}m")
            print(f"   Avg Sim Frequency: {step/total_time:.1f}Hz")
            print(f"   Avg Mirror Frequency: {self.mirror_count/total_time:.1f}Hz")
            print(f"   Commands Sent: {self.mirror_count}")
            print(f"{'='*70}\n")

        return {
            "success": True,
            "steps": step,
            "reward": episode_reward,
            "distance": episode_distance,
            "duration": total_time,
            "mirror_count": self.mirror_count
        }

    def detach_servos(self):
        """Send command to detach servos (safe shutdown)."""
        print(f"\n🔌 Detaching servos...")
        try:
            self.action_socket.send_multipart([b"cmd", b"detach_servos\n"])
            time.sleep(0.5)
        except:
            pass

    def close(self):
        """Clean shutdown."""
        self.detach_servos()
        self.vec_env.close()
        self.action_socket.close()


def main():
    parser = argparse.ArgumentParser(
        description="Mirror MuJoCo simulation to physical RealANT robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run M3 (best model) with visualization
  python simulation_mirroring.py \\
      --model ../../experiments/M3_dr_seed42/best_model/best_model.zip \\
      --vec-normalize ../../experiments/M3_dr_seed42/vec_normalize.pkl \\
      --render \\
      --episodes 3

  # Run without rendering (faster)
  python simulation_mirroring.py \\
      --model ../../experiments/M3_dr_seed42/best_model/best_model.zip \\
      --vec-normalize ../../experiments/M3_dr_seed42/vec_normalize.pkl \\
      --no-render \\
      --episodes 5

  # Custom frequencies
  python simulation_mirroring.py \\
      --model ../../experiments/M3_dr_seed42/best_model/best_model.zip \\
      --vec-normalize ../../experiments/M3_dr_seed42/vec_normalize.pkl \\
      --sim-freq 100 \\
      --mirror-freq 50
        """
    )

    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--vec-normalize", required=True, help="Path to VecNormalize (.pkl)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--episode-length", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Show MuJoCo visualization")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Hide visualization")
    parser.add_argument("--sim-freq", type=float, default=50.0, help="Simulation frequency (Hz)")
    parser.add_argument("--mirror-freq", type=float, default=15.0, help="Robot mirror frequency (Hz) - reduced to prevent communication overload")
    parser.add_argument("--action-port", type=int, default=3001, help="ZeroMQ action port (ant_server listens here)")
    parser.add_argument("--no-safety", dest="safety", action="store_false", help="Disable safety limits")
    parser.add_argument("--rate-limit", type=int, default=50, help="Max change in servo position per step (0 to disable).")

    parser.add_argument("--action-smoothing", type=float, default=0.3, help="EMA filter alpha (0-1). Lower is smoother. 1 to disable.")
    parser.add_argument("--action-scale", type=float, default=0.6, help="Scale factor for actions (0.5=half speed). Prevents motor overload.")
    parser.add_argument("--no-pid", action="store_true", help="Disable the software PID controller.")

    parser.set_defaults(render=True, safety=True)
    args = parser.parse_args()

    try:
        # Initialize mirroring system
        bridge = SimulationMirroringBridge(
            model_path=args.model,
            vec_normalize_path=args.vec_normalize,
            render=args.render,
            mirror_freq=args.mirror_freq,
            sim_freq=args.sim_freq,
            action_port=args.action_port,
            safety_limits=args.safety,
            action_smoothing=args.action_smoothing,
            action_scale=args.action_scale,
            rate_limit=args.rate_limit,
            use_pid=not args.no_pid
        )

        # Run episodes
        results = []

        for ep in range(args.episodes):
            print(f"\n{'#'*70}")
            print(f"# Episode {ep+1}/{args.episodes}")
            print(f"{'#'*70}")

            result = bridge.run_episode(
                episode_length=args.episode_length,
                verbose=True
            )

            if result["success"]:
                results.append(result)
            else:
                print(f"❌ Episode failed!")
                break

            if ep < args.episodes - 1:
                print(f"\n⏸️  Reset robot to starting position, then press Enter...")
                input()

        # Final summary
        if results:
            print(f"\n{'='*70}")
            print(f"🏁 Mirroring Complete!")
            print(f"{'='*70}")
            rewards = [r["reward"] for r in results]
            distances = [r["distance"] for r in results]
            print(f"   Episodes Completed: {len(results)}")
            print(f"   Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"   Avg Distance: {np.mean(distances):.3f} ± {np.std(distances):.3f}m")
            print(f"   Best Distance: {np.max(distances):.3f}m")
            print(f"{'='*70}\n")

        # Cleanup
        bridge.close()
        return 0

    except KeyboardInterrupt:
        print(f"\n\n⏹️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
