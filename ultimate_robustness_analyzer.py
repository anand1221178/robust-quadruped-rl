
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import cv2
import argparse
from pathlib import Path
import time

# Assuming these custom wrappers are in the PYTHONPATH
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper
import realant_sim

class UltimateRobustnessAnalyzer:
    """
    A 2-pass video evaluation tool to create sciency, data-rich analysis videos
    for any trained quadruped model.
    """
    def __init__(self, model_path, vec_normalize_path, output_path, config):
        self.model_path = Path(model_path)
        self.vec_normalize_path = Path(vec_normalize_path)
        self.output_path = Path(output_path)
        self.config = config
        self.model_name = self.model_path.parent.name

        print(">> Initializing Analyzer...")
        self._load_model_and_env()

    def _load_model_and_env(self):
        """Loads the environment and the trained model."""
        print(f">> Loading model: {self.model_path}")
        print(f">> Loading VecNormalize stats: {self.vec_normalize_path}")

        # Create a non-normalized environment for data collection
        self.eval_env_raw = DummyVecEnv([lambda: SuccessRewardWrapper(gym.make('RealAntMujoco-v0'))])
        
        # Load the normalization stats but DON'T wrap the environment yet
        self.vec_normalize = VecNormalize.load(self.vec_normalize_path, self.eval_env_raw)
        self.vec_normalize.training = False
        self.vec_normalize.norm_reward = False

        # Load the PPO model
        self.model = PPO.load(self.model_path, env=self.eval_env_raw) # Env will be wrapped later
        print(">> Model and environment loaded successfully.")

    def run_analysis(self, test_scenarios):
        """Runs the analysis for a given set of test scenarios."""
        for name, params in test_scenarios.items():
            print(f"\n--- Running Scenario: {name} ---")
            
            # --- Pass 1: Data Collection ---
            print(">> Pass 1: Collecting data without rendering...")
            trajectory_data = self._collect_trajectory(params)
            print(f">> Pass 1 Complete. Collected {len(trajectory_data['actions'])} steps.")

            # --- Pass 2: Video Rendering ---
            print(">> Pass 2: Rendering video with sci-fi HUD...")
            self._render_video(trajectory_data, name, params)
            print(f">> Pass 2 Complete. Video saved.")

    def _collect_trajectory(self, params):
        """Pass 1: Run simulation without rendering to get pure performance data."""
        
        # This is a bit of a hack to apply DR to a VecNormalized env
        # We manually create the wrapper chain
        def make_env():
            env = gym.make('RealAntMujoco-v0')
            env = SuccessRewardWrapper(env)
            # Apply domain randomization based on scenario params
            if params.get('use_dr', False):
                dr_params = {
                    'joint_dropout_prob': params.get('joint_failure_prob', 0.0),
                    'max_dropped_joints': len(params.get('failed_joints', [])),
                    'forced_dropped_joints': params.get('failed_joints', None),
                    'failure_mode': 'torque_multiplier',
                    'joint_torque_multiplier': params.get('k_value', 1.0),
                    'sensor_noise_std': params.get('sensor_noise', 0.0)
                }
                env = DomainRandomizationWrapper(env, dr_params)
            return env

        env = DummyVecEnv([make_env])
        env = VecNormalize.load(self.vec_normalize_path, env)
        env.training = False
        env.norm_reward = False

        obs = env.reset()
        
        data = {
            "actions": [], "observations": [], "rewards": [], 
            "infos": [], "positions": [], "velocities": []
        }

        for _ in range(self.config['max_steps']):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            data["actions"].append(action)
            data["observations"].append(obs)
            data["rewards"].append(reward)
            data["infos"].append(info[0])
            # Correctly access position data for RealAnt
            position = env.envs[0].unwrapped.data.qpos
            data["positions"].append((position[0], position[1], position[2]))
            data["velocities"].append(info[0].get('velocity', 0.0))
            
            if done:
                break
        
        env.close()
        return data

    def _render_video(self, trajectory_data, scenario_name, params):
        """Pass 2: Replay actions and render video with a cool HUD."""
        
        # Create a renderable environment
        render_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_file = self.output_path / f"{self.model_name}_{scenario_name}_{timestamp}.mp4"
        video = cv2.VideoWriter(str(video_file), fourcc, self.config['fps'], 
                                (self.config['width'], self.config['height']))

        obs = render_env.reset()

        # HUD elements
        hud = HUD(self.config, self.model_name, scenario_name, params)

        for i in range(len(trajectory_data["actions"])):
            # Force the recorded action
            action = trajectory_data["actions"][i][0]
            render_env.step(action)
            
            # Get frame
            frame = render_env.render()
            
            # Update HUD with current data
            hud.update(
                velocity=trajectory_data['velocities'][i],
                position=trajectory_data['positions'][i],
                info=trajectory_data['infos'][i],
                step=i
            )

            # Draw HUD on the frame
            final_frame = hud.draw(frame)
            video.write(final_frame)

        video.release()
        render_env.close()
        print(f"\n>>> Analysis complete! Video saved to {video_file} <<<")


class HUD:
    """Heads-Up Display manager for rendering sci-fi overlays."""
    def __init__(self, config, model_name, scenario_name, params):
        # Config
        self.width = config['width']
        self.height = config['height']
        self.hud_width = 400
        
        # Colors and Fonts
        self.bg_color = (15, 20, 25)
        self.primary_color = (0, 255, 200) # Cyan
        self.secondary_color = (255, 100, 0)
        self.text_color = (220, 220, 220)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_mono = cv2.FONT_HERSHEY_PLAIN

        # Static Info
        self.model_name = model_name
        self.scenario_name = scenario_name
        self.params = params

        # Dynamic data storage for plots
        self.vel_history = [0] * 150
        self.pos_history = []

    def update(self, velocity, position, info, step):
        self.velocity = velocity
        self.position = position
        self.info = info
        self.step = step
        
        # Update velocity history for plot
        self.vel_history.pop(0)
        self.vel_history.append(velocity)
        
        # Update position history
        self.pos_history.append((position[0], position[1]))

    def draw(self, frame):
        """Draws the HUD onto the main video frame."""
        # Resize frame to fit next to HUD
        render_frame = cv2.resize(frame, (self.width - self.hud_width, self.height))
        
        # Create HUD canvas
        hud_canvas = np.full((self.height, self.hud_width, 3), self.bg_color, dtype=np.uint8)

        # --- Draw elements ---
        self._draw_text(hud_canvas)
        self._draw_velocity_plot(hud_canvas, 200)
        self._draw_joint_status(hud_canvas, 380)
        self._draw_trajectory_map(hud_canvas, 560)
        
        # Combine render frame and HUD
        final_frame = np.concatenate((render_frame, hud_canvas), axis=1)
        return final_frame

    def _draw_text(self, canvas):
        cv2.putText(canvas, "ROBUSTNESS ANALYSIS", (10, 30), self.font, 0.7, self.primary_color, 2)
        cv2.line(canvas, (10, 35), (390, 35), self.primary_color, 1)

        info_y = 60
        cv2.putText(canvas, f"MODEL: {self.model_name}", (10, info_y), self.font_mono, 1.2, self.text_color, 1)
        cv2.putText(canvas, f"SCENARIO: {self.scenario_name}", (10, info_y + 20), self.font_mono, 1.2, self.text_color, 1)
        cv2.putText(canvas, f"STEP: {self.step}", (10, info_y + 40), self.font_mono, 1.2, self.text_color, 1)
        cv2.putText(canvas, f"VELOCITY: {self.velocity:.3f} m/s", (10, info_y + 60), self.font_mono, 1.2, self.primary_color, 1)
        
        failed_joints = self.info.get('dropped_joints', [])
        k_val = self.info.get('joint_torque_multiplier', 1.0)
        noise = self.params.get('sensor_noise', 0.0)
        
        k_color = self.secondary_color if k_val < 1.0 else self.text_color
        cv2.putText(canvas, f"K-VALUE: {k_val:.2f}", (10, info_y + 90), self.font_mono, 1.2, k_color, 1)
        
        noise_color = self.secondary_color if noise > 0 else self.text_color
        cv2.putText(canvas, f"NOISE: {noise:.3f}", (120, info_y + 90), self.font_mono, 1.2, noise_color, 1)

        failures_color = self.secondary_color if failed_joints else self.text_color
        cv2.putText(canvas, f"FAILURES: {len(failed_joints)}", (250, info_y + 90), self.font_mono, 1.2, failures_color, 1)

    def _draw_velocity_plot(self, canvas, y_offset):
        cv2.putText(canvas, "Velocity (m/s)", (10, y_offset - 10), self.font_mono, 1.2, self.text_color, 1)
        plot_h, plot_w = 100, 380
        cv2.rectangle(canvas, (10, y_offset), (10 + plot_w, y_offset + plot_h), self.primary_color, 1)

        max_vel = max(abs(v) for v in self.vel_history) if self.vel_history else 1.0
        max_vel = max(max_vel, 0.5) # Set a minimum scale

        for i in range(1, len(self.vel_history)):
            y1 = int((plot_h / 2) - (self.vel_history[i-1] / max_vel) * (plot_h / 2))
            y2 = int((plot_h / 2) - (self.vel_history[i] / max_vel) * (plot_h / 2))
            x1 = int((i - 1) * (plot_w / len(self.vel_history)))
            x2 = int(i * (plot_w / len(self.vel_history)))
            cv2.line(canvas, (10 + x1, y_offset + y1), (10 + x2, y_offset + y2), self.primary_color, 1)
        # Zero line
        cv2.line(canvas, (10, y_offset + plot_h // 2), (10 + plot_w, y_offset + plot_h // 2), (255, 255, 255, 50), 1)

    def _draw_joint_status(self, canvas, y_offset):
        cv2.putText(canvas, "Joint Status", (10, y_offset - 10), self.font_mono, 1.2, self.text_color, 1)
        
        failed_joints = self.info.get('dropped_joints', [])
        k_val = self.info.get('joint_torque_multiplier', 1.0)

        # Simple quadruped layout
        body_pos = [(80, y_offset + 40), (280, y_offset + 40), (80, y_offset + 100), (280, y_offset + 100)]
        joint_labels = ['F.L', 'F.R', 'R.L', 'R.R']
        
        for i in range(4): # 4 legs
            hip_joint_idx = i * 2
            ankle_joint_idx = i * 2 + 1
            
            # Hip status
            hip_color = self.text_color
            if hip_joint_idx in failed_joints:
                hip_color = self.secondary_color if k_val > 0 else (0, 0, 255) # Red for dead
            
            # Ankle status
            ankle_color = self.text_color
            if ankle_joint_idx in failed_joints:
                ankle_color = self.secondary_color if k_val > 0 else (0, 0, 255)

            cv2.rectangle(canvas, (body_pos[i][0], body_pos[i][1]), (body_pos[i][0] + 40, body_pos[i][1] + 20), hip_color, -1)
            cv2.rectangle(canvas, (body_pos[i][0], body_pos[i][1] + 25), (body_pos[i][0] + 40, body_pos[i][1] + 45), ankle_color, -1)
            cv2.putText(canvas, joint_labels[i], (body_pos[i][0] + 5, body_pos[i][1] - 5), self.font_mono, 1, self.text_color, 1)

    def _draw_trajectory_map(self, canvas, y_offset):
        cv2.putText(canvas, "Trajectory (Top-Down)", (10, y_offset - 10), self.font_mono, 1.2, self.text_color, 1)
        map_h, map_w = 150, 380
        cv2.rectangle(canvas, (10, y_offset), (10 + map_w, y_offset + map_h), self.primary_color, 1)

        if not self.pos_history: return

        positions = np.array(self.pos_history)
        
        # Auto-scale
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        
        scale_x = (map_w - 20) / max(1.0, max_x - min_x)
        scale_y = (map_h - 20) / max(1.0, max_y - min_y)
        scale = min(scale_x, scale_y)

        for i in range(1, len(self.pos_history)):
            p1 = positions[i-1]
            p2 = positions[i]
            
            x1 = int(10 + 10 + (p1[1] - min_y) * scale)
            y1 = int(y_offset + 10 + (p1[0] - min_x) * scale)
            x2 = int(10 + 10 + (p2[1] - min_y) * scale)
            y2 = int(y_offset + 10 + (p2[0] - min_x) * scale)
            
            cv2.line(canvas, (x1, y1), (x2, y2), self.primary_color, 2)


if __name__ == '__main__':
    # Hardcoded model path for easier testing
    model_path_str = 'done/ppo_sr2l_forward_m7gtjtpa/final_model.zip'
    print(f"-- Using hardcoded model: {model_path_str} --")
    model_path = Path(model_path_str)
    vec_normalize_path = model_path.parent / 'vec_normalize.pkl'
    
    if not model_path.exists() or not vec_normalize_path.exists():
        raise FileNotFoundError("Could not find model .zip or vec_normalize.pkl in the specified directory.")

    # --- CONFIGURATION ---
    VIDEO_CONFIG = {
        'width': 1280 + 400, # Render width + HUD width
        'height': 720,
        'fps': 30,
        'max_steps': 1000,
    }

    # --- DEFINE YOUR TEST SCENARIOS HERE ---
    TEST_SCENARIOS = {
        "Baseline": {
            "use_dr": False,
        },
        "Sensor_Noise_0.02": {
            "use_dr": True,
            "sensor_noise": 0.02,
        },
        "Hip_Failure_k0.5": {
            "use_dr": True,
            "failed_joints": [0, 2], # Both front hips
            "k_value": 0.5,
        },
        "Ankle_Failure_k0.0": {
            "use_dr": True,
            "failed_joints": [1, 7], # One front, one rear ankle
            "k_value": 0.0, # Completely dead
        },
    }

    analyzer = UltimateRobustnessAnalyzer(
        model_path=str(model_path),
        vec_normalize_path=str(vec_normalize_path),
        output_path=model_path.parent,
        config=VIDEO_CONFIG
    )

    analyzer.run_analysis(TEST_SCENARIOS)
