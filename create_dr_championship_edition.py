#!/usr/bin/env python3
"""
🏆 DR CHAMPIONSHIP EDITION - JOINT FAILURE ROBUSTNESS 🏆
REAL joint failure testing with forced failures and epic visualizations
FIXED: Proper top-down camera implementation
"""
import sys
sys.path.append('src')

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import cv2
import os
from datetime import datetime
import json

class DRChampionRecorder:
    """DR Championship Video Creator with FORCED joint failures"""
    
    def __init__(self):
        self.frame_size = (1920, 1080)
        self.fps = 60  # High quality 60fps
        self.baseline_performance = None  # Will be calculated from no-failure performance
        
        # Test duration per level
        self.frames_per_level = 1800  # 30 seconds per level at 60fps (EXTENDED for rotation!)
        
        # Joint failure test levels (FORCED failures!)
        self.failure_levels = [
            {'name': 'NO FAILURES', 'joints': [], 'prob': 0.0},
            {'name': 'HIP_1 LOCKED', 'joints': ['hip_1'], 'prob': 1.0},
            {'name': 'ANKLE_1 LOCKED', 'joints': ['ankle_1'], 'prob': 1.0},
            {'name': 'HIP_2 LOCKED', 'joints': ['hip_2'], 'prob': 1.0},
            {'name': 'ANKLE_2 LOCKED', 'joints': ['ankle_2'], 'prob': 1.0},
            {'name': 'HIP_3 LOCKED', 'joints': ['hip_3'], 'prob': 1.0},
            {'name': 'ANKLE_3 LOCKED', 'joints': ['ankle_3'], 'prob': 1.0},
            {'name': 'HIP_4 LOCKED', 'joints': ['hip_4'], 'prob': 1.0},
            {'name': 'ANKLE_4 LOCKED', 'joints': ['ankle_4'], 'prob': 1.0},
        ]
        
        # Epic moments
        self.epic_moments = {
            'NO FAILURES': "🎯 BASELINE PERFORMANCE",
            'HIP_1 LOCKED': "🦵 FRONT LEFT LEG CHALLENGE", 
            'ANKLE_1 LOCKED': "🦶 FRONT LEFT FOOT LOCKED",
            'HIP_4 LOCKED': "🦵 REAR RIGHT LEG CHALLENGE",
            'ANKLE_4 LOCKED': "🦶 REAR RIGHT FOOT LOCKED"
        }
        
        # Colors (BGR format for OpenCV)
        self.colors = {
            'champion': (0, 215, 255),    # Gold
            'excellent': (0, 255, 0),     # Green
            'good': (0, 255, 255),        # Yellow  
            'challenge': (0, 165, 255),   # Orange
            'extreme': (0, 0, 255),       # Red
            'background': (0, 0, 0),      # Black
            'text': (255, 255, 255),      # White
            'failure': (255, 0, 255)      # Magenta for failed joints
        }
        
        # Joint name mapping (MuJoCo joint names to display names)
        self.joint_display_names = {
            'hip_1': 'Front Left Hip',
            'ankle_1': 'Front Left Ankle', 
            'hip_2': 'Front Right Hip',
            'ankle_2': 'Front Right Ankle',
            'hip_3': 'Rear Left Hip', 
            'ankle_3': 'Rear Left Ankle',
            'hip_4': 'Rear Right Hip',
            'ankle_4': 'Rear Right Ankle'
        }
    
    def setup_topdown_camera(self, env):
        """Properly configure top-down camera view"""
        try:
            # Direct MuJoCo model access - most reliable method
            if hasattr(env.envs[0], 'unwrapped'):
                unwrapped = env.envs[0].unwrapped
                if hasattr(unwrapped, 'model') and hasattr(unwrapped, 'data'):
                    model = unwrapped.model
                    data = unwrapped.data
                    
                    # Set camera position directly above the ant looking straight down
                    # Position: X=0 (centered), Y=-8 (back), Z=8 (height)
                    model.cam_pos[0] = [0, -8, 8]
                    
                    # Quaternion for looking straight down with proper orientation
                    # This quaternion represents a rotation to look down from above
                    model.cam_quat[0] = [0.7071, 0.7071, 0, 0]  # 90 degrees around X-axis
                    
                    print("  ✅ Top-down camera configured via MuJoCo model")
                    return True
                    
            # Alternative: Access the MuJoCo renderer if available
            if hasattr(env.envs[0], 'mujoco_renderer'):
                renderer = env.envs[0].mujoco_renderer
                if hasattr(renderer, 'viewer'):
                    viewer = renderer.viewer
                    
                    # Set camera to look straight down
                    viewer.cam.lookat[0] = 0.0  # X position
                    viewer.cam.lookat[1] = 0.0  # Y position  
                    viewer.cam.lookat[2] = 0.0  # Z position (ground level)
                    
                    viewer.cam.distance = 8.0   # Height above ground
                    viewer.cam.elevation = -90  # Look straight down
                    viewer.cam.azimuth = 90    # Rotation angle
                    
                    print("  ✅ Top-down camera set via viewer")
                    return True
                    
        except Exception as e:
            print(f"  ⚠️ Camera setup attempt failed: {e}")
            
        return False
    
    def get_topdown_frame(self, env):
        """Get frame with explicit top-down camera parameters"""
        # Just use standard render since we've set the camera position
        frame = env.envs[0].render()
        if frame is not None:
            frame = cv2.resize(frame, self.frame_size)
        return frame
    
    def force_joint_failure(self, env, joint_names):
        """FORCE specific joints to fail by locking their actions"""
        if not joint_names:
            return None
            
        # Direct action index mapping for RealAnt
        joint_to_action_index = {
            'hip_1': 0,    # Front left hip
            'ankle_1': 1,  # Front left ankle
            'hip_2': 2,    # Front right hip 
            'ankle_2': 3,  # Front right ankle
            'hip_3': 4,    # Rear left hip
            'ankle_3': 5,  # Rear left ankle
            'hip_4': 6,    # Rear right hip
            'ankle_4': 7,  # Rear right ankle
        }
        
        joint_indices = []
        for joint_name in joint_names:
            if joint_name in joint_to_action_index:
                joint_indices.append(joint_to_action_index[joint_name])
            else:
                print(f"Warning: Joint {joint_name} not found in mapping")
                
        return joint_indices
    
    def apply_joint_failure(self, action, failed_joint_indices):
        """Lock specific joint actions to 0 (failure)"""
        if not failed_joint_indices:
            return action
            
        action_copy = action.copy()
        for joint_idx in failed_joint_indices:
            if joint_idx < len(action_copy[0]):
                action_copy[0][joint_idx] = 0.0  # Lock joint
                
        return action_copy
    
    def get_performance_color(self, retention_pct):
        """Get color based on retention percentage"""
        if retention_pct >= 95:
            return self.colors['champion']  # Gold for excellent
        elif retention_pct >= 80:
            return self.colors['excellent']  # Green for good
        elif retention_pct >= 60:
            return self.colors['good']       # Yellow for ok
        elif retention_pct >= 40:
            return self.colors['challenge']  # Orange for challenging
        else:
            return self.colors['extreme']    # Red for extreme
    
    def create_dr_overlay(self, frame, failure_level, current_velocity, episode_progress, 
                         current_distance, retention_pct, failed_joints):
        """Create DR championship overlay graphics"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent background for HUD
        cv2.rectangle(overlay, (0, 0), (w, 220), self.colors['background'], -1)
        cv2.rectangle(overlay, (0, h-180), (w, h), self.colors['background'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Championship title
        title = "DR CHAMPIONSHIP: TOP-DOWN JOINT FAILURE TEST"
        cv2.putText(frame, title, (50, 50), font, 1.2, self.colors['champion'], 3)
        
        # Current challenge level
        challenge_text = f"FAILURE MODE: {failure_level['name']}"
        if failure_level['name'] in self.epic_moments:
            challenge_text += f" - {self.epic_moments[failure_level['name']]}"
        cv2.putText(frame, challenge_text, (50, 90), font, 0.8, self.colors['text'], 2)
        
        # Failed joints display
        if failed_joints:
            joints_text = "FAILED JOINTS: " + ", ".join([self.joint_display_names.get(j, j) for j in failed_joints])
            cv2.putText(frame, joints_text, (50, 130), font, 0.7, self.colors['failure'], 2)
        else:
            cv2.putText(frame, "FAILED JOINTS: None (Normal Operation)", (50, 130), font, 0.7, self.colors['excellent'], 2)
        
        # Performance metrics dashboard
        metrics_y = 170
        cv2.putText(frame, f"VELOCITY: {current_velocity:.3f} m/s", 
                   (50, metrics_y), font, 0.7, self.colors['text'], 2)
        
        cv2.putText(frame, f"DISTANCE: {current_distance:.1f}m", 
                   (300, metrics_y), font, 0.7, self.colors['text'], 2)
        
        # Retention percentage (big and colorful)
        retention_color = self.get_performance_color(retention_pct)
        cv2.putText(frame, f"RETENTION: {retention_pct:.1f}%", 
                   (550, metrics_y), font, 0.8, retention_color, 2)
        
        # Failure severity indicator
        if not failed_joints:
            severity_text = "NORMAL OPERATION"
            severity_color = self.colors['excellent']
        else:
            severity_text = f"JOINT FAILURE ACTIVE"
            severity_color = self.colors['failure']
        
        cv2.putText(frame, severity_text, (850, metrics_y), font, 0.7, severity_color, 2)
        
        # Progress bar
        bar_width = 400
        bar_height = 20
        bar_x = w - bar_width - 50
        bar_y = h - 120
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Progress fill
        progress_width = int(bar_width * episode_progress)
        progress_color = self.get_performance_color(retention_pct)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                     progress_color, -1)
        
        # Progress text
        cv2.putText(frame, f"TEST PROGRESS: {episode_progress*100:.0f}%", 
                   (bar_x, bar_y - 10), font, 0.6, self.colors['text'], 2)
        
        # Special annotations for epic moments
        if failure_level['name'] in ['HIP_1 LOCKED', 'ANKLE_4 LOCKED']:
            cv2.putText(frame, "CRITICAL JOINT FAILURE TEST", 
                       (50, h - 80), font, 1.0, self.colors['failure'], 3)
        elif retention_pct > 80 and failed_joints:
            cv2.putText(frame, "EXCELLENT ROBUSTNESS!", 
                       (50, h - 80), font, 1.0, self.colors['champion'], 3)
        
        # Failure visualizer
        self.draw_failure_visualizer(frame, failed_joints)
        
        return frame
    
    def draw_failure_visualizer(self, frame, failed_joints):
        """Draw visual representation of failed joints"""
        h, w = frame.shape[:2]
        
        viz_x = w - 300
        viz_y = 80
        viz_width = 250
        viz_height = 100
        
        # Background
        cv2.rectangle(frame, (viz_x, viz_y), (viz_x + viz_width, viz_y + viz_height), 
                     (50, 50, 50), -1)
        
        # Draw simple robot representation
        robot_x = viz_x + viz_width // 2
        robot_y = viz_y + viz_height // 2
        
        # Body
        cv2.rectangle(frame, (robot_x - 30, robot_y - 15), (robot_x + 30, robot_y + 15), 
                     self.colors['text'], 2)
        
        # Legs (as lines)
        leg_positions = {
            'hip_1': (robot_x - 20, robot_y - 15, robot_x - 35, robot_y - 35),   # Front left
            'ankle_1': (robot_x - 35, robot_y - 35, robot_x - 35, robot_y - 50),
            'hip_2': (robot_x + 20, robot_y - 15, robot_x + 35, robot_y - 35),   # Front right  
            'ankle_2': (robot_x + 35, robot_y - 35, robot_x + 35, robot_y - 50),
            'hip_3': (robot_x - 20, robot_y + 15, robot_x - 35, robot_y + 35),   # Rear left
            'ankle_3': (robot_x - 35, robot_y + 35, robot_x - 35, robot_y + 50),
            'hip_4': (robot_x + 20, robot_y + 15, robot_x + 35, robot_y + 35),   # Rear right
            'ankle_4': (robot_x + 35, robot_y + 35, robot_x + 35, robot_y + 50),
        }
        
        # Draw legs
        for joint, (x1, y1, x2, y2) in leg_positions.items():
            color = self.colors['failure'] if joint in failed_joints else self.colors['excellent']
            thickness = 4 if joint in failed_joints else 2
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label
        cv2.putText(frame, "JOINT STATUS", (viz_x, viz_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def record_dr_championship(self, model_path, vec_path, output_path):
        """Record the complete DR championship demonstration"""
        print("=" * 80)
        print("DR CHAMPIONSHIP EDITION - TOP-DOWN JOINT FAILURE ROBUSTNESS")
        print("=" * 80)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)
        
        total_frames = 0
        performance_data = []
        
        for level_idx, failure_level in enumerate(self.failure_levels):
            print(f"\nTesting failure level {level_idx+1}/{len(self.failure_levels)}: {failure_level['name']}")
            print(f"Target frames for this level: {self.frames_per_level}")
            
            # Create environment (clean, no DR wrapper) with EXTENDED episodes
            def make_env():
                # Create base environment without automatic TimeLimit wrapper
                base_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array', disable_env_checker=True)

                # Remove any existing TimeLimit wrappers
                while isinstance(base_env, TimeLimit):
                    base_env = base_env.env

                # Add our own TimeLimit with extended episode length
                env = TimeLimit(base_env, max_episode_steps=2500)  # 2500 steps = ~41.7 seconds
                env = SuccessRewardWrapper(env)

                print(f"  ✅ Environment configured with 2500 max steps (extended for rotation testing)")
                return env
            
            env = DummyVecEnv([make_env])
            
            # Load model and normalization
            try:
                env = VecNormalize.load(vec_path, env)
                env.training = False
                env.norm_reward = False
                print("  VecNormalize loaded")
            except:
                print("  No VecNormalize")
            
            model = PPO.load(model_path)
            print("  Model loaded")
            
            # Setup top-down camera
            self.setup_topdown_camera(env)
            
            # Get joint indices for failure injection
            failed_joint_indices = self.force_joint_failure(env, failure_level['joints'])
            
            # Record episode with current failure level
            obs = env.reset()
            
            # Additional camera setup after reset
            self.setup_topdown_camera(env)
            
            positions = []
            rewards = []
            frames_this_level = 0

            # DELAYED LOCKING: Let robot establish proper gait and momentum first!
            DELAYED_LOCKING_STEPS = 120  # Lock joints after 120 steps (2 full seconds)
            joint_lock_active = False

            print(f"  Starting episode with {failure_level['name']}...")
            if failed_joint_indices:
                print(f"  Forcing failure on joints: {failure_level['joints']} (indices: {failed_joint_indices})")
                print(f"  🚀 DELAYED LOCKING: Joints will lock after {DELAYED_LOCKING_STEPS} steps (2 seconds)")

            # Track if we need to continue across episode boundaries
            episode_resets = 0
            continuous_positions = []  # Track positions across resets

            for step in range(2000):  # Extended max steps for rotation test
                if frames_this_level >= self.frames_per_level:
                    break

                # Get action from model
                action, _ = model.predict(obs, deterministic=True)

                # DELAYED LOCKING: Only force joints to fail after initial steps
                if failed_joint_indices and step >= DELAYED_LOCKING_STEPS:
                    if not joint_lock_active:
                        joint_lock_active = True
                        print(f"    💥 JOINTS LOCKED at step {step} - Movement first, then adaptation!")
                    action = self.apply_joint_failure(action, failed_joint_indices)

                obs, reward, done, info = env.step(action)

                # Track metrics
                x_pos = env.envs[0].unwrapped.data.qpos[0]

                # Handle position tracking across resets
                if episode_resets > 0 and len(positions) > 0:
                    # Add offset from previous episode
                    x_pos += continuous_positions[-1]

                positions.append(x_pos)
                rewards.append(reward[0])

                # Get frame with top-down view
                frame = self.get_topdown_frame(env)

                if frame is not None:
                    # Calculate current metrics
                    if len(positions) >= 2:
                        current_distance = positions[-1] - positions[0]
                        time_elapsed = len(positions) * 0.05
                        current_velocity = current_distance / time_elapsed
                    else:
                        current_distance = 0
                        current_velocity = 0

                    # Calculate retention
                    if level_idx == 0 and self.baseline_performance is None:
                        # First run, no baseline yet
                        retention_pct = 100.0
                    else:
                        retention_pct = (current_velocity / self.baseline_performance) * 100 if self.baseline_performance and self.baseline_performance > 0 else 0

                    episode_progress = frames_this_level / self.frames_per_level

                    # Apply DR overlay
                    frame = self.create_dr_overlay(
                        frame, failure_level, current_velocity, episode_progress,
                        current_distance, retention_pct, failure_level['joints']
                    )

                    video_writer.write(frame)
                    frames_this_level += 1
                    total_frames += 1

                # Handle episode reset but continue recording
                if done[0] and frames_this_level < self.frames_per_level:
                    print(f"    Episode {episode_resets + 1} ended at step {step}, continuing...")
                    # Save continuous position for offset
                    if len(positions) > 0:
                        continuous_positions.append(positions[-1])

                    # Reset environment but keep recording
                    obs = env.reset()
                    # Re-setup camera after reset
                    self.setup_topdown_camera(env)
                    episode_resets += 1

                    # Don't break, continue recording!
            
            # Store performance data
            if len(positions) >= 2:
                initial_x = positions[0]
                # Use last valid position
                final_x = positions[-1]
                final_distance = final_x - initial_x

                # Use actual frames recorded for time calculation
                time_taken = frames_this_level / self.fps
                final_velocity = final_distance / time_taken if time_taken > 0 else 0

                print(f"    Total distance traveled: {final_distance:.2f}m over {time_taken:.1f}s")
                
                # Set baseline from first (no-failure) run
                if self.baseline_performance is None and failure_level['name'] == 'NO FAILURES':
                    self.baseline_performance = final_velocity
                    print(f"  ✅ Baseline performance set: {self.baseline_performance:.3f} m/s")
                
                retention = (final_velocity / self.baseline_performance) * 100 if self.baseline_performance and self.baseline_performance > 0 else 0
                
                performance_data.append({
                    'failure_mode': failure_level['name'],
                    'failed_joints': failure_level['joints'],
                    'velocity': final_velocity,
                    'retention_pct': retention,
                    'distance': final_distance,
                    'frames_recorded': frames_this_level
                })
                
                print(f"  Final metrics: {final_velocity:.3f} m/s ({retention:.1f}% retention)")
            
            env.close()
        
        video_writer.release()
        
        # Save performance data
        with open(output_path.replace('.mp4', '_performance.json'), 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        print(f"\nDR CHAMPIONSHIP VIDEO COMPLETE!")
        print(f"Video: {output_path}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/self.fps:.1f} seconds")
        
        return output_path, performance_data

def main():
    """Create the DR Championship Edition video"""

    # V7.7E ULTRA SPEED - THE ORIGINAL CHAMPION!
    # Testing with FIXED evaluation (2500 steps) to see its TRUE potential
    # The ONLY model to achieve positive ankle_4 retention!

    model_name = "v7_7e_ultra_speed_jtfwl2qf"
    model_path = "/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/final_model.zip"
    vec_path = "/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl"

    # Available V7.7 models to test:
    # v7_7_speed_champion_3yjaqdrp      - Speed bonuses for walking >0.12 m/s with failures
    # v7_7b_joint_aware_z92bswuk        - Joint-specific penalties (hips > ankles)
    # v7_7c_ankle4_specialist_b8qotdf1  - Ankle-4 specialist (rear right ankle focus)
    # v7_7d_progressive_mastery_2iwcmhoh - Progressive difficulty with ultra-dense rewards
    # v7_7e_ultra_speed_jtfwl2qf        - Multi-tier speed bonuses (0.10, 0.12, 0.15 m/s)
    # v7_7f_combined_ultimate_uhbvlz02  - Everything combined for ultimate robustness
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean model name for output
    clean_name = model_name.rsplit('_', 1)[0]  # Remove run ID
    output_path = f"videos/{clean_name}_CHAMPION_{timestamp}.mp4"
    
    os.makedirs("videos", exist_ok=True)
    
    recorder = DRChampionRecorder()
    
    video_path, performance_data = recorder.record_dr_championship(
        model_path, vec_path, output_path
    )
    
    print(f"\nDR CHAMPIONSHIP EDITION COMPLETE!")
    print(f"Video: {video_path}")
    
    print(f"\nCHAMPIONSHIP HIGHLIGHTS:")
    for data in performance_data:
        failure_mode = data['failure_mode']
        retention = data['retention_pct']
        frames = data.get('frames_recorded', 0)
        joints = ', '.join(data['failed_joints']) if data['failed_joints'] else 'None'
        print(f"  {failure_mode}: {retention:.1f}% retention, Failed: {joints} ({frames} frames)")

if __name__ == "__main__":
    main()
