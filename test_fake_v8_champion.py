#!/usr/bin/env python3
"""
🏆 FAKE V8 CONSERVATIVE CHAMPION TEST 🏆
Testing the 60M step "super-baseline" model that trained without V8 curriculum
This could be our new baseline ceiling for robustness comparisons!
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
import cv2
import json
from pathlib import Path
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

# Championship color scheme
class ChampionshipColors:
    GOLD = (0, 215, 255)      # Championship gold
    WHITE = (255, 255, 255)   # Pure white text
    BLUE = (255, 100, 50)     # Electric blue
    GREEN = (50, 255, 50)     # Success green
    RED = (50, 50, 255)       # Alert red
    BLACK = (0, 0, 0)         # Background
    SILVER = (192, 192, 192)  # Silver accents
    PURPLE = (255, 50, 255)   # V8 purple

class FakeV8Champion:
    def __init__(self):
        # Updated paths for the fake V8 Conservative model
        self.model_path = "experiments/final_model.zip"
        self.vecnorm_path = "experiments/vec_normalize.pkl"

        # Video settings
        self.width = 1920
        self.height = 1080
        self.fps = 60

        # Performance tracking
        self.positions = []
        self.velocities = []
        self.rewards = []
        self.timesteps = []

        print("🚀 FAKE V8 CONSERVATIVE CHAMPION - Testing 60M step super-baseline...")
        print(f"   Model: {self.model_path}")
        print(f"   VecNormalize: {self.vecnorm_path}")

    def load_model(self):
        """Load the fake V8 Conservative champion model"""
        try:
            # Create environment (same as training - just SuccessRewardWrapper)
            def make_env():
                env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
                # Add SuccessRewardWrapper (what it actually trained with)
                env = SuccessRewardWrapper(env)
                return env

            # Create vectorized environment
            env = DummyVecEnv([make_env])

            # Load VecNormalize
            env = VecNormalize.load(self.vecnorm_path, env)
            env.training = False  # Evaluation mode
            env.norm_reward = False  # Don't normalize rewards during eval

            # Load model
            model = PPO.load(self.model_path)

            self.env = env
            self.model = model

            print("✅ FAKE V8 Conservative model loaded successfully!")
            print("   This model trained for 60M steps (6x standard baseline)")
            print("   Should show MUCH higher performance than 0.224 m/s standard baseline")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def create_championship_overlay(self, frame, episode_data, step, total_steps):
        """Create epic championship overlay for fake V8 Conservative"""

        # Current metrics
        current_pos = episode_data['positions'][-1] if episode_data['positions'] else 0.0
        current_vel = episode_data['velocities'][-1] if episode_data['velocities'] else 0.0
        current_reward = episode_data['rewards'][-1] if episode_data['rewards'] else 0.0

        # Calculate episode metrics
        if len(episode_data['positions']) > 10:
            total_distance = abs(current_pos)
            avg_velocity = np.mean(episode_data['velocities'][-100:])  # Last 5 seconds
            total_reward = sum(episode_data['rewards'])
        else:
            total_distance = 0.0
            avg_velocity = 0.0
            total_reward = 0.0

        # Create overlay
        overlay = frame.copy()

        # Championship header
        header_height = 120
        cv2.rectangle(overlay, (0, 0), (self.width, header_height), ChampionshipColors.BLACK, -1)
        cv2.rectangle(overlay, (0, 0), (self.width, header_height), ChampionshipColors.PURPLE, 3)

        # Title
        title_text = "FAKE V8 CONSERVATIVE CHAMPION - 60M STEP SUPER-BASELINE"
        cv2.putText(overlay, title_text, (50, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, ChampionshipColors.PURPLE, 2)

        # Subtitle
        subtitle = f"Extended Training Model - 6x Longer Than Standard Baseline - Pure Speed Optimization"
        cv2.putText(overlay, subtitle, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ChampionshipColors.WHITE, 2)

        # Progress bar
        progress = step / total_steps
        bar_width = self.width - 100
        bar_height = 20
        bar_y = 90

        # Progress bar background
        cv2.rectangle(overlay, (50, bar_y), (50 + bar_width, bar_y + bar_height), ChampionshipColors.BLACK, -1)
        cv2.rectangle(overlay, (50, bar_y), (50 + bar_width, bar_y + bar_height), ChampionshipColors.WHITE, 2)

        # Progress fill
        fill_width = int(bar_width * progress)
        cv2.rectangle(overlay, (50, bar_y), (50 + fill_width, bar_y + bar_height), ChampionshipColors.PURPLE, -1)

        # Step counter
        step_text = f"Step: {step}/{total_steps}"
        cv2.putText(overlay, step_text, (self.width - 200, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ChampionshipColors.WHITE, 2)

        # Performance metrics panel
        panel_x = 50
        panel_y = 150
        panel_width = 500
        panel_height = 300

        # Panel background
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     ChampionshipColors.BLACK, -1)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     ChampionshipColors.PURPLE, 3)

        # Panel title
        cv2.putText(overlay, "SUPER-BASELINE METRICS", (panel_x + 20, panel_y + 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, ChampionshipColors.PURPLE, 2)

        # Metrics
        metrics = [
            ("Current Velocity", f"{current_vel:.3f} m/s", ChampionshipColors.GREEN),
            ("Average Velocity", f"{avg_velocity:.3f} m/s", ChampionshipColors.WHITE),
            ("Total Distance", f"{total_distance:.2f} m", ChampionshipColors.BLUE),
            ("Current Reward", f"{current_reward:.0f}", ChampionshipColors.WHITE),
            ("Episode Reward", f"{total_reward:.0f}", ChampionshipColors.WHITE),
            ("X Position", f"{current_pos:.2f} m", ChampionshipColors.SILVER),
            ("vs Standard", f"{'Higher!' if avg_velocity > 0.224 else 'Check'}", ChampionshipColors.GREEN if avg_velocity > 0.224 else ChampionshipColors.RED),
        ]

        for i, (label, value, color) in enumerate(metrics):
            y_pos = panel_y + 60 + (i * 35)
            cv2.putText(overlay, f"{label}:", (panel_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ChampionshipColors.WHITE, 1)
            cv2.putText(overlay, value, (panel_x + 250, y_pos),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

        # Championship stats panel
        stats_x = self.width - 450
        stats_y = 150
        stats_width = 400
        stats_height = 300

        # Stats panel background
        cv2.rectangle(overlay, (stats_x, stats_y), (stats_x + stats_width, stats_y + stats_height),
                     ChampionshipColors.BLACK, -1)
        cv2.rectangle(overlay, (stats_x, stats_y), (stats_x + stats_width, stats_y + stats_height),
                     ChampionshipColors.PURPLE, 3)

        # Stats title
        cv2.putText(overlay, "EXTENDED TRAINING STATS", (stats_x + 20, stats_y + 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, ChampionshipColors.PURPLE, 2)

        # Championship statistics
        stats = [
            ("Training Steps", "60M Steps", ChampionshipColors.PURPLE),
            ("Training Time", "23+ Hours", ChampionshipColors.WHITE),
            ("vs Standard", "6x Longer", ChampionshipColors.GREEN),
            ("Architecture", "PPO + [64,128]", ChampionshipColors.WHITE),
            ("Environment", "RealAnt + Success", ChampionshipColors.WHITE),
            ("Expected Speed", ">0.30 m/s?", ChampionshipColors.GREEN),
            ("Robustness", "ZERO Training", ChampionshipColors.RED),
        ]

        for i, (label, value, color) in enumerate(stats):
            y_pos = stats_y + 60 + (i * 35)
            cv2.putText(overlay, f"{label}:", (stats_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ChampionshipColors.WHITE, 1)
            cv2.putText(overlay, value, (stats_x + 200, y_pos),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

        # Velocity graph (mini real-time plot)
        if len(episode_data['velocities']) > 1:
            graph_x = 50
            graph_y = self.height - 200
            graph_width = 400
            graph_height = 100

            # Graph background
            cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height),
                         ChampionshipColors.BLACK, -1)
            cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height),
                         ChampionshipColors.PURPLE, 2)

            # Graph title
            cv2.putText(overlay, "VELOCITY PROFILE", (graph_x + 10, graph_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ChampionshipColors.PURPLE, 2)

            # Plot velocities
            recent_vels = episode_data['velocities'][-100:] if len(episode_data['velocities']) > 100 else episode_data['velocities']
            if len(recent_vels) > 1:
                max_vel = max(recent_vels) if max(recent_vels) > 0 else 1.0

                for i in range(1, len(recent_vels)):
                    x1 = graph_x + int((i-1) * graph_width / len(recent_vels))
                    y1 = graph_y + graph_height - int(recent_vels[i-1] / max_vel * graph_height)
                    x2 = graph_x + int(i * graph_width / len(recent_vels))
                    y2 = graph_y + graph_height - int(recent_vels[i] / max_vel * graph_height)
                    cv2.line(overlay, (x1, y1), (x2, y2), ChampionshipColors.GREEN, 2)

        # Championship footer
        footer_y = self.height - 80
        cv2.rectangle(overlay, (0, footer_y), (self.width, self.height), ChampionshipColors.BLACK, -1)
        cv2.rectangle(overlay, (0, footer_y), (self.width, self.height), ChampionshipColors.PURPLE, 3)

        footer_text = f"FAKE V8 CONSERVATIVE - 60M Step Extended Training - Testing Super-Baseline Performance"
        cv2.putText(overlay, footer_text, (50, footer_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, ChampionshipColors.PURPLE, 2)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(overlay, f"Generated: {timestamp}", (50, footer_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ChampionshipColors.WHITE, 1)

        return overlay

    def record_championship_demo(self, episodes=1, max_steps=1000):
        """Record epic championship demonstration of fake V8 Conservative"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"FAKE_V8_CONSERVATIVE_CHAMPION_{timestamp}.mp4"

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, self.fps, (self.width, self.height))

        print(f"🎬 Recording FAKE V8 Conservative championship demonstration...")
        print(f"   Episodes: {episodes}")
        print(f"   Max steps per episode: {max_steps}")
        print(f"   Video: {video_filename}")
        print(f"   Expectation: Should beat standard 0.224 m/s baseline!")

        all_episodes_data = []

        for episode in range(episodes):
            print(f"\n🎥 Recording Episode {episode + 1}/{episodes}")

            obs = self.env.reset()
            episode_data = {
                'positions': [],
                'velocities': [],
                'rewards': [],
                'timesteps': []
            }

            for step in range(max_steps):
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, done, info = self.env.step(action)

                # Extract position and calculate velocity
                if hasattr(self.env.envs[0].unwrapped, 'data'):
                    x_pos = float(self.env.envs[0].unwrapped.data.qpos[0])

                    # Calculate velocity
                    if len(episode_data['positions']) > 0:
                        dt = 0.05  # 50ms timestep
                        velocity = (x_pos - episode_data['positions'][-1]) / dt
                    else:
                        velocity = 0.0

                    episode_data['positions'].append(x_pos)
                    episode_data['velocities'].append(velocity)
                    episode_data['rewards'].append(float(reward[0]))
                    episode_data['timesteps'].append(step)

                # Render frame
                frame = self.env.render(mode='rgb_array')

                # Resize to target resolution
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Add championship overlay
                overlayed_frame = self.create_championship_overlay(frame, episode_data, step + 1, max_steps)

                # Write frame
                video_writer.write(overlayed_frame)

                # Progress update
                if (step + 1) % 100 == 0:
                    current_vel = episode_data['velocities'][-1] if episode_data['velocities'] else 0.0
                    current_pos = episode_data['positions'][-1] if episode_data['positions'] else 0.0
                    vs_baseline = "🚀 FASTER!" if current_vel > 0.224 else "🤔 Check"
                    print(f"   Step {step + 1}: {current_vel:.3f} m/s, {current_pos:.2f} m {vs_baseline}")

                if done[0]:
                    print(f"   Episode completed at step {step + 1}")
                    break

            # Store episode data
            all_episodes_data.append(episode_data)

            # Episode summary
            if episode_data['positions']:
                final_distance = abs(episode_data['positions'][-1])
                avg_velocity = np.mean(episode_data['velocities'])
                total_reward = sum(episode_data['rewards'])

                # Performance comparison
                vs_baseline_perf = avg_velocity / 0.224 * 100  # Percentage of baseline
                performance_verdict = "🏆 SUPER-BASELINE!" if avg_velocity > 0.30 else "🚀 BETTER!" if avg_velocity > 0.224 else "🤔 INVESTIGATE"

                print(f"\n📊 Episode {episode + 1} Results:")
                print(f"   📏 Distance: {final_distance:.2f} m")
                print(f"   🏃 Average Velocity: {avg_velocity:.3f} m/s")
                print(f"   📈 vs Baseline (0.224): {vs_baseline_perf:.1f}% {performance_verdict}")
                print(f"   🎯 Total Reward: {total_reward:.0f}")

        video_writer.release()

        # Save performance data
        performance_data = {
            'model_type': 'fake_v8_conservative',
            'model_path': self.model_path,
            'vecnorm_path': self.vecnorm_path,
            'training_steps': '60M',
            'training_hours': '23+',
            'baseline_comparison': 0.224,
            'timestamp': timestamp,
            'episodes_data': all_episodes_data,
            'video_filename': video_filename,
            'summary': {
                'episodes': episodes,
                'max_steps': max_steps,
                'total_frames': sum(len(ep['positions']) for ep in all_episodes_data)
            }
        }

        performance_filename = f"FAKE_V8_CONSERVATIVE_CHAMPION_{timestamp}_performance.json"

        # Convert numpy arrays to lists for JSON serialization
        for episode_data in performance_data['episodes_data']:
            for key in episode_data:
                if isinstance(episode_data[key], list):
                    episode_data[key] = [float(x) for x in episode_data[key]]

        with open(performance_filename, 'w') as f:
            json.dump(performance_data, f, indent=2)

        print(f"\n🏆 FAKE V8 Conservative championship test complete!")
        print(f"   🎬 Video: {video_filename}")
        print(f"   📊 Performance data: {performance_filename}")
        print(f"   📁 File size: {Path(video_filename).stat().st_size / (1024*1024):.1f} MB")

        # Final verdict
        if all_episodes_data:
            final_avg_vel = np.mean([np.mean(ep['velocities']) for ep in all_episodes_data])
            if final_avg_vel > 0.30:
                print(f"\n🎉 VERDICT: SUPER-BASELINE CONFIRMED! ({final_avg_vel:.3f} m/s)")
                print("   This should become your new baseline for robustness comparisons!")
            elif final_avg_vel > 0.224:
                print(f"\n✅ VERDICT: IMPROVED BASELINE! ({final_avg_vel:.3f} m/s)")
                print("   Extended training paid off - better than standard baseline!")
            else:
                print(f"\n🤔 VERDICT: INVESTIGATE ({final_avg_vel:.3f} m/s)")
                print("   Performance similar to standard baseline - check training logs")

        return video_filename, performance_filename

def main():
    """Test the fake V8 Conservative super-baseline model"""

    print("🚀 FAKE V8 CONSERVATIVE CHAMPION TEST")
    print("Testing 60M step 'super-baseline' model for performance ceiling")
    print("=" * 70)

    # Create champion tester
    champion = FakeV8Champion()

    # Load model
    champion.load_model()

    # Record championship demo
    video_file, performance_file = champion.record_championship_demo(
        episodes=1,
        max_steps=1000  # ~50 seconds at 20 FPS environment
    )

    print("\n🎊 FAKE V8 CONSERVATIVE TEST COMPLETE!")
    print(f"🎬 Video: {video_file}")
    print(f"📊 Data: {performance_file}")
    print("\n   Ready to determine if this becomes your new baseline!")

if __name__ == "__main__":
    main()