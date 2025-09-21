
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import cv2
from pathlib import Path
import time

# Assuming these custom wrappers are in the PYTHONPATH
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper
import realant_sim

class GrandChampionAnalyzer:
    """
    Combines the best of both approaches:
    - Creates ONE championship video sequencing through multiple scenarios.
    - Uses a robust 2-pass system for accurate metrics.
    - Features an advanced, data-rich HUD.
    - Is easily configurable for any model and any set of scenarios.
    """
    def __init__(self, model_path, vec_normalize_path, output_path, config):
        self.model_path = Path(model_path)
        self.vec_normalize_path = Path(vec_normalize_path)
        self.output_path = Path(output_path)
        self.config = config
        self.model_name = self.model_path.parent.name

        print(">> Initializing Grand Champion Analyzer...")
        self._load_model_and_env()

    def _load_model_and_env(self):
        print(f">> Loading model: {self.model_path}")
        self.model = PPO.load(self.model_path)
        print(">> Model loaded successfully.")

    def run_championship(self, test_scenarios):
        """Runs all scenarios, then renders them into a single video."""
        
        # --- PASS 1: DATA COLLECTION ---
        print("\n--- PASS 1: COLLECTING TRAJECTORY DATA FOR ALL SCENARIOS ---")
        all_trajectory_data = []
        for name, params in test_scenarios.items():
            print(f"  > Collecting data for scenario: {name}")
            trajectory = self._collect_trajectory(params)
            all_trajectory_data.append({"name": name, "params": params, "data": trajectory})
            print(f"    ...done. Collected {len(trajectory['actions'])} steps.")

        # --- PASS 2: VIDEO RENDERING ---
        print("\n--- PASS 2: RENDERING CHAMPIONSHIP VIDEO ---")
        self._render_championship_video(all_trajectory_data)

    def _collect_trajectory(self, params):
        """Runs a single scenario without rendering to get pure performance data."""
        def make_env():
            env = gym.make('RealAntMujoco-v0')
            env = SuccessRewardWrapper(env)
            if params.get('use_dr', False):
                dr_params = {
                    'joint_dropout_prob': 1.0 if params.get('failed_joints') else 0.0,
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
        trajectory = {"actions": [], "infos": [], "qpos": []}

        for _ in range(self.config['max_steps']):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)

            trajectory["actions"].append(action)
            trajectory["infos"].append(info[0])
            trajectory["qpos"].append(env.envs[0].unwrapped.data.qpos.copy())
            
            if done:
                break
        
        env.close()
        return trajectory

    def _render_championship_video(self, all_trajectory_data):
        """Renders all collected trajectories into a single video file."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_file = self.output_path / f"GRAND_CHAMPION_{self.model_name}_{timestamp}.mp4"
        video = cv2.VideoWriter(str(video_file), fourcc, self.config['fps'], 
                                (self.config['width'], self.config['height']))

        for i, segment in enumerate(all_trajectory_data):
            scenario_name = segment["name"]
            params = segment["params"]
            trajectory = segment["data"]
            print(f"  > Rendering segment {i+1}/{len(all_trajectory_data)}: {scenario_name}")

            # 1. Add Title Card
            title_card = self._create_title_card(scenario_name, params)
            for _ in range(self.config['fps'] * 3): # 3-second hold
                video.write(title_card)

            # 2. Render the actual segment
            render_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
            render_env.reset()
            hud = HUD(self.config, self.model_name, scenario_name, params)

            for step_idx in range(len(trajectory["actions"])):
                action = trajectory["actions"][step_idx][0]
                render_env.step(action)
                frame = render_env.render()
                
                hud.update(
                    qpos_history=trajectory["qpos"][:step_idx+1],
                    info=trajectory["infos"][step_idx],
                    step=step_idx
                )

                final_frame = hud.draw(frame)
                video.write(final_frame)
            
            render_env.close()

        video.release()
        print(f"\n\n>>> GRAND CHAMPIONSHIP video complete! Saved to {video_file} <<<")

    def _create_title_card(self, scenario_name, params):
        """Creates a frame to introduce the next scenario."""
        card = np.full((self.config['height'], self.config['width'], 3), (15, 20, 25), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(card, "NEXT SCENARIO", (50, 100), font, 1.5, (0, 255, 200), 3)
        cv2.putText(card, scenario_name, (50, 200), font, 2.5, (255, 255, 255), 5)

        y_offset = 300
        for key, value in params.items():
            text = f"- {key}: {value}"
            cv2.putText(card, text, (50, y_offset), font, 1.2, (200, 200, 200), 2)
            y_offset += 50

        return card

class HUD:
    def __init__(self, config, model_name, scenario_name, params):
        self.width = config['width']
        self.height = config['height']
        self.hud_width = 400
        self.bg_color = (15, 20, 25)
        self.primary_color = (0, 255, 200)
        self.secondary_color = (255, 100, 0)
        self.text_color = (220, 220, 220)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_mono = cv2.FONT_HERSHEY_PLAIN
        self.model_name = model_name
        self.scenario_name = scenario_name
        self.params = params
        self.vel_history = [0] * 150

    def update(self, qpos_history, info, step):
        self.qpos_history = qpos_history
        self.info = info
        self.step = step

        # CORRECTED VELOCITY CALCULATION
        if len(self.qpos_history) >= 2:
            start_pos = self.qpos_history[0]
            current_pos = self.qpos_history[-1]
            distance = current_pos[0] - start_pos[0]
            time_elapsed = len(self.qpos_history) * 0.05 # Assumes 20Hz control
            self.velocity = distance / time_elapsed if time_elapsed > 0 else 0.0
        else:
            self.velocity = 0.0
        
        self.vel_history.pop(0)
        self.vel_history.append(self.velocity)

    def draw(self, frame):
        render_frame = cv2.resize(frame, (self.width - self.hud_width, self.height))
        hud_canvas = np.full((self.height, self.hud_width, 3), self.bg_color, dtype=np.uint8)
        self._draw_text(hud_canvas)
        self._draw_velocity_plot(hud_canvas, 200)
        self._draw_joint_status(hud_canvas, 380)
        self._draw_trajectory_map(hud_canvas, 560)
        return np.concatenate((render_frame, hud_canvas), axis=1)

    def _draw_text(self, canvas):
        cv2.putText(canvas, "GRAND CHAMPION ANALYSIS", (10, 30), self.font, 0.7, self.primary_color, 2)
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
        max_vel = max(max_vel, 0.5)
        for i in range(1, len(self.vel_history)):
            y1 = int((plot_h / 2) - (self.vel_history[i-1] / max_vel) * (plot_h / 2))
            y2 = int((plot_h / 2) - (self.vel_history[i] / max_vel) * (plot_h / 2))
            x1 = int((i - 1) * (plot_w / len(self.vel_history)))
            x2 = int(i * (plot_w / len(self.vel_history)))
            cv2.line(canvas, (10 + x1, y_offset + y1), (10 + x2, y_offset + y2), self.primary_color, 1)
        cv2.line(canvas, (10, y_offset + plot_h // 2), (10 + plot_w, y_offset + plot_h // 2), (255, 255, 255, 50), 1)

    def _draw_joint_status(self, canvas, y_offset):
        cv2.putText(canvas, "Joint Status", (10, y_offset - 10), self.font_mono, 1.2, self.text_color, 1)
        failed_joints = self.info.get('dropped_joints', [])
        k_val = self.info.get('joint_torque_multiplier', 1.0)
        body_pos = [(80, y_offset + 40), (280, y_offset + 40), (80, y_offset + 100), (280, y_offset + 100)]
        joint_labels = ['F.L', 'F.R', 'R.L', 'R.R']
        for i in range(4):
            hip_joint_idx = i * 2
            ankle_joint_idx = i * 2 + 1
            hip_color = self.text_color
            if hip_joint_idx in failed_joints:
                hip_color = self.secondary_color if k_val > 0 else (0, 0, 255)
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
        positions = np.array([q[:2] for q in self.qpos_history])
        if len(positions) < 2: return
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        scale_x = (map_w - 20) / max(1.0, max_x - min_x)
        scale_y = (map_h - 20) / max(1.0, max_y - min_y)
        scale = min(scale_x, scale_y)
        for i in range(1, len(positions)):
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
        raise FileNotFoundError(f"Could not find model .zip or vec_normalize.pkl in {model_path.parent}")

    VIDEO_CONFIG = {
        'width': 1280 + 400, 'height': 720, 'fps': 30, 'max_steps': 500, # Shorter segments for showcase
    }

    TEST_SCENARIOS = {
        "Baseline": {"use_dr": False},
        "Training_Noise": {"use_dr": True, "sensor_noise": 0.01},
        "Extreme_Noise": {"use_dr": True, "sensor_noise": 0.1},
        "Partial_Hip_Failure": {"use_dr": True, "failed_joints": [0, 2], "k_value": 0.5},
        "Total_Ankle_Failure": {"use_dr": True, "failed_joints": [5, 7], "k_value": 0.0},
    }

    analyzer = GrandChampionAnalyzer(
        model_path=str(model_path),
        vec_normalize_path=str(vec_normalize_path),
        output_path=model_path.parent,
        config=VIDEO_CONFIG
    )

    analyzer.run_championship(TEST_SCENARIOS)
