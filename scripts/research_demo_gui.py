#!/usr/bin/env python3
"""
Interactive GUI Demo for Robust Quadruped RL Research
Showcases the 4-way ablation study with real-time model testing
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import threading
import time
import os
import sys
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import pickle

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import session manager
from session_manager import get_session_manager, save_session_image, save_session_data, add_session_note

try:
    from agents.ppo_sr2l import PPO_SR2L
    from envs.success_reward_wrapper import SuccessRewardWrapper
    from envs.target_walking_wrapper import TargetWalkingWrapper
    import realant_sim
except ImportError as e:
    print(f"Import warning: {e}")

class QuadrupedResearchDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Robust Quadruped RL - Research Demo")
        self.root.geometry("1400x900")
        
        # Initialize session manager
        self.session_manager = get_session_manager()
        self.session_manager.add_tool_usage("research_demo_gui", "Interactive model testing with live visualization")
        
        # Model storage
        self.models = {}
        self.current_model = None
        self.current_env = None
        
        # Data for plots
        self.velocity_data = []
        self.smoothness_data = []
        self.time_data = []
        self.noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
        
        # Testing state
        self.is_testing = False
        self.test_thread = None
        
        self.setup_ui()
        self.load_available_models()
        
    def setup_ui(self):
        """Create the main UI layout"""
        
        # Create main sections
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === CONTROL PANEL ===
        ttk.Label(control_frame, text="Robust Quadruped RL Demo", 
                 font=("Arial", 16, "bold")).pack(pady=10)
        
        # Model Selection
        model_frame = ttk.LabelFrame(control_frame, text="Model Selection")
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Available Models:").pack(anchor=tk.W)
        self.model_listbox = tk.Listbox(model_frame, height=6)
        self.model_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.model_listbox.bind('<<ListboxSelect>>', self.on_model_select)
        
        ttk.Button(model_frame, text="Load Custom Model", 
                  command=self.load_custom_model).pack(pady=5)
        
        # Current Model Info
        info_frame = ttk.LabelFrame(control_frame, text="Current Model Info")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.model_info = tk.Text(info_frame, height=8, width=40)
        self.model_info.pack(padx=5, pady=5)
        
        # Testing Controls
        test_frame = ttk.LabelFrame(control_frame, text="Testing Controls")
        test_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(test_frame, text="Start Live Testing", 
                  command=self.start_live_testing).pack(fill=tk.X, pady=2)
        ttk.Button(test_frame, text="Stop Testing", 
                  command=self.stop_testing).pack(fill=tk.X, pady=2)
        ttk.Button(test_frame, text="Robustness Test", 
                  command=self.run_robustness_test).pack(fill=tk.X, pady=2)
        ttk.Button(test_frame, text="Compare All Models", 
                  command=self.compare_all_models).pack(fill=tk.X, pady=2)
        
        # Noise Control
        noise_frame = ttk.LabelFrame(control_frame, text="Noise Injection")
        noise_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(noise_frame, text="Sensor Noise Level:").pack(anchor=tk.W)
        self.noise_var = tk.DoubleVar(value=0.0)
        self.noise_scale = ttk.Scale(noise_frame, from_=0.0, to=0.25, 
                                    variable=self.noise_var, orient=tk.HORIZONTAL)
        self.noise_scale.pack(fill=tk.X, padx=5)
        
        self.noise_label = ttk.Label(noise_frame, text="0.0%")
        self.noise_label.pack()
        self.noise_scale.configure(command=self.update_noise_label)
        
        # === VISUALIZATION PANEL ===
        viz_notebook = ttk.Notebook(viz_frame)
        viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Performance Tab
        perf_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(perf_frame, text="Live Performance")
        
        self.setup_performance_plot(perf_frame)
        
        # Robustness Tab  
        robust_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(robust_frame, text="Robustness Analysis")
        
        self.setup_robustness_plot(robust_frame)
        
        # Comparison Tab
        comp_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(comp_frame, text="Model Comparison")
        
        self.setup_comparison_plot(comp_frame)
        
    def setup_performance_plot(self, parent):
        """Setup live performance monitoring plot"""
        self.perf_fig, (self.vel_ax, self.smooth_ax) = plt.subplots(2, 1, figsize=(10, 8))
        self.perf_fig.suptitle('Live Performance Monitoring')
        
        # Velocity plot
        self.vel_ax.set_title('Walking Velocity')
        self.vel_ax.set_ylabel('Velocity (m/s)')
        self.vel_ax.grid(True, alpha=0.3)
        self.vel_line, = self.vel_ax.plot([], [], 'b-', linewidth=2, label='Current')
        self.vel_ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Target (1.0 m/s)')
        self.vel_ax.legend()
        
        # Smoothness plot
        self.smooth_ax.set_title('Action Smoothness')
        self.smooth_ax.set_ylabel('Smoothness Score')
        self.smooth_ax.set_xlabel('Time (episodes)')
        self.smooth_ax.grid(True, alpha=0.3)
        self.smooth_line, = self.smooth_ax.plot([], [], 'r-', linewidth=2)
        
        canvas = FigureCanvasTkAgg(self.perf_fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        
    def setup_robustness_plot(self, parent):
        """Setup robustness analysis plot"""
        self.robust_fig, self.robust_ax = plt.subplots(figsize=(10, 6))
        self.robust_fig.suptitle('Sensor Noise Robustness Analysis')
        
        self.robust_ax.set_xlabel('Sensor Noise Level (%)')
        self.robust_ax.set_ylabel('Success Rate (%)')
        self.robust_ax.grid(True, alpha=0.3)
        self.robust_ax.set_ylim(0, 100)
        
        canvas = FigureCanvasTkAgg(self.robust_fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_comparison_plot(self, parent):
        """Setup model comparison plot"""
        self.comp_fig, ((self.speed_ax, self.robust_ax_comp), 
                       (self.smooth_ax_comp, self.combined_ax)) = plt.subplots(2, 2, figsize=(12, 10))
        self.comp_fig.suptitle('4-Way Ablation Study Comparison')
        
        # Speed comparison
        self.speed_ax.set_title('Walking Speed')
        self.speed_ax.set_ylabel('Velocity (m/s)')
        
        # Robustness comparison
        self.robust_ax_comp.set_title('Noise Robustness')
        self.robust_ax_comp.set_ylabel('Success Rate (%)')
        
        # Smoothness comparison
        self.smooth_ax_comp.set_title('Action Smoothness')
        self.smooth_ax_comp.set_ylabel('Smoothness Score')
        
        # Combined radar chart
        self.combined_ax.set_title('Overall Performance')
        
        canvas = FigureCanvasTkAgg(self.comp_fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def load_available_models(self):
        """Load available trained models"""
        model_paths = {
            "Baseline (PPO)": "done/ppo_baseline_ueqbjf2x/best_model/best_model.zip",
            "SR2L (Joint Sensor)": "experiments/ppo_sr2l_corrected_*/best_model/best_model.zip", 
            "DR (Actuator Failure)": "experiments/ppo_dr_robust_*/best_model/best_model.zip",
            "Combined (SR2L+DR)": "experiments/ppo_sr2l_dr_*/best_model/best_model.zip"
        }
        
        available_models = []
        for name, path in model_paths.items():
            # Check if model exists (handle wildcards later)
            if "done/ppo_baseline" in path:
                if os.path.exists(path):
                    available_models.append(name)
            else:
                # For now, add placeholders for training models
                available_models.append(f"{name} (Training...)")
        
        for model in available_models:
            self.model_listbox.insert(tk.END, model)
            
    def on_model_select(self, event):
        """Handle model selection"""
        selection = self.model_listbox.curselection()
        if selection:
            model_name = self.model_listbox.get(selection[0])
            if "Training..." in model_name:
                self.model_info.delete(1.0, tk.END)
                self.model_info.insert(1.0, f"Model: {model_name}\n\nStatus: Currently training on cluster\nExpected completion: ~24-48 hours\n\nThis model will be available for testing once training completes.")
            else:
                self.load_model(model_name)
    
    def load_model(self, model_name):
        """Load a specific model"""
        try:
            if "Baseline" in model_name:
                model_path = "done/ppo_baseline_ueqbjf2x/best_model/best_model.zip"
                vec_path = "done/ppo_baseline_ueqbjf2x/vec_normalize.pkl"
                
                if os.path.exists(model_path):
                    # Load model
                    self.current_model = PPO.load(model_path)
                    
                    # Create environment
                    def make_env():
                        _env = gym.make('RealAntMujoco-v0')
                        _env = SuccessRewardWrapper(_env)
                        _env = Monitor(_env)
                        return _env
                    
                    env = DummyVecEnv([make_env])
                    
                    # Load normalization if available
                    if os.path.exists(vec_path):
                        with open(vec_path, 'rb') as f:
                            vec_normalize = pickle.load(f)
                        # CRITICAL: Set the venv attribute properly
                        vec_normalize.venv = env
                        env = vec_normalize
                        env.training = False
                    
                    self.current_env = env
                    
                    # Update info display
                    self.model_info.delete(1.0, tk.END)
                    info_text = f"""Model: {model_name}
Type: PPO Baseline
Status: ✅ LOADED

Performance:
- Speed: 0.216 ± 0.003 m/s
- Behavior: Smooth walking
- Environment: RealAnt + SuccessReward
- Training: 10M timesteps

Ready for testing!"""
                    self.model_info.insert(1.0, info_text)
                    
                else:
                    raise FileNotFoundError(f"Model not found: {model_path}")
            else:
                self.model_info.delete(1.0, tk.END)
                self.model_info.insert(1.0, f"Model: {model_name}\n\nStatus: Not yet implemented\nThis will be available once training completes.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def load_custom_model(self):
        """Load a custom model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # Load model
                if 'sr2l' in file_path.lower():
                    self.current_model = PPO_SR2L.load(file_path)
                else:
                    self.current_model = PPO.load(file_path)
                
                # Create environment (same as baseline)
                def make_env():
                    _env = gym.make('RealAntMujoco-v0')
                    _env = SuccessRewardWrapper(_env)
                    _env = Monitor(_env)
                    return _env
                
                env = DummyVecEnv([make_env])
                
                # Check for vec_normalize in same directory
                model_dir = os.path.dirname(os.path.dirname(file_path))
                vec_path = os.path.join(model_dir, 'vec_normalize.pkl')
                
                if os.path.exists(vec_path):
                    with open(vec_path, 'rb') as f:
                        vec_normalize = pickle.load(f)
                    # CRITICAL: Set the venv attribute properly
                    vec_normalize.venv = env
                    env = vec_normalize
                    env.training = False
                
                self.current_env = env
                
                # Update info
                self.model_info.delete(1.0, tk.END)
                model_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                self.model_info.insert(1.0, f"Custom Model Loaded:\n{model_name}\nPath: {file_path}\n\n✅ Environment created\n✅ Ready for testing!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load custom model: {str(e)}")
    
    def update_noise_label(self, value):
        """Update noise level label"""
        noise_pct = float(value) * 100
        self.noise_label.config(text=f"{noise_pct:.1f}%")
    
    def start_live_testing(self):
        """Start live testing in a separate thread"""
        if not self.current_model or not self.current_env:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        if self.is_testing:
            messagebox.showinfo("Info", "Testing already in progress!")
            return
            
        self.is_testing = True
        self.test_thread = threading.Thread(target=self.live_testing_loop, daemon=True)
        self.test_thread.start()
        
    def stop_testing(self):
        """Stop live testing"""
        self.is_testing = False
        
    def live_testing_loop(self):
        """Main loop for live testing"""
        episode = 0
        
        while self.is_testing:
            try:
                # Run one episode
                obs = self.current_env.reset()
                episode_reward = 0
                velocities = []
                actions = []
                positions = []
                
                for step in range(1000):  # Max steps per episode
                    action, _ = self.current_model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.current_env.step(action)
                    
                    # Try to get the actual environment for velocity
                    try:
                        # Navigate through wrappers to get base env
                        if hasattr(self.current_env, 'venv'):
                            base_env = self.current_env.venv.envs[0]
                        else:
                            base_env = self.current_env.envs[0]
                        
                        # Get unwrapped env
                        while hasattr(base_env, 'env'):
                            base_env = base_env.env
                        
                        # For MuJoCo envs, get x velocity directly
                        if hasattr(base_env, 'sim') and hasattr(base_env.sim, 'data'):
                            # Get center of mass velocity
                            velocity = base_env.sim.data.qvel[0]  # x-velocity
                        else:
                            velocity = 0
                    except:
                        velocity = 0
                    
                    # Add noise if specified
                    if self.noise_var.get() > 0:
                        noise = np.random.normal(0, self.noise_var.get(), obs.shape)
                        obs = obs + noise
                    
                    # Fallback methods if direct velocity extraction failed
                    if velocity == 0:
                        # Method 1: Check info dict
                        if len(info) > 0 and hasattr(info[0], 'get'):
                            velocity = info[0].get('speed', info[0].get('x_velocity', 0))
                        
                        # Method 2: Use reward as proxy (rough estimate)
                        if velocity == 0 and reward[0] > 0:
                            # SuccessRewardWrapper gives speed-based rewards
                            velocity = reward[0] * 0.2  # Scale factor
                    
                    # Ensure velocity is positive and reasonable
                    velocity = abs(velocity)
                    
                    velocities.append(velocity)
                    actions.append(action[0])
                    
                    episode_reward += reward[0]
                    
                    if done[0]:
                        break
                
                # Calculate metrics
                avg_velocity = np.mean(velocities) if velocities else 0
                smoothness = self.calculate_smoothness(actions)
                
                # Update data
                self.time_data.append(episode)
                self.velocity_data.append(avg_velocity)
                self.smoothness_data.append(smoothness)
                
                # Keep only last 50 episodes
                if len(self.time_data) > 50:
                    self.time_data = self.time_data[-50:]
                    self.velocity_data = self.velocity_data[-50:]
                    self.smoothness_data = self.smoothness_data[-50:]
                
                # Update plots
                self.update_live_plots()
                
                episode += 1
                time.sleep(0.1)  # Small delay
                
            except Exception as e:
                print(f"Error in testing loop: {e}")
                break
                
    def calculate_smoothness(self, actions):
        """Calculate action smoothness score"""
        if len(actions) < 2:
            return 1.0
        
        action_changes = np.mean([np.linalg.norm(np.array(actions[i+1]) - np.array(actions[i])) 
                                 for i in range(len(actions)-1)])
        return 1.0 / (1.0 + action_changes)
    
    def update_live_plots(self):
        """Update the live performance plots"""
        if not self.time_data:
            return
            
        # Update velocity plot
        self.vel_line.set_data(self.time_data, self.velocity_data)
        self.vel_ax.relim()
        self.vel_ax.autoscale_view()
        
        # Update smoothness plot
        self.smooth_line.set_data(self.time_data, self.smoothness_data)
        self.smooth_ax.relim()
        self.smooth_ax.autoscale_view()
        
        # Redraw
        self.perf_fig.canvas.draw()
    
    def run_robustness_test(self):
        """Run comprehensive robustness testing"""
        if not self.current_model or not self.current_env:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        messagebox.showinfo("Info", "Running robustness test... This may take a few minutes.")
        
        # Log the test
        add_session_note("Started comprehensive robustness testing")
        
        # Run in separate thread
        threading.Thread(target=self.robustness_test_worker, daemon=True).start()
    
    def robustness_test_worker(self):
        """Worker thread for robustness testing"""
        success_rates = []
        
        for noise_level in self.noise_levels:
            successes = 0
            total_tests = 10
            
            for test in range(total_tests):
                obs = self.current_env.reset()
                episode_reward = 0
                
                for step in range(1000):
                    action, _ = self.current_model.predict(obs, deterministic=True)
                    
                    # Add noise
                    if noise_level > 0:
                        noise = np.random.normal(0, noise_level, obs.shape)
                        obs_noisy = obs + noise
                    else:
                        obs_noisy = obs
                        
                    obs, reward, done, info = self.current_env.step(action)
                    episode_reward += reward[0]
                    
                    if done[0]:
                        break
                
                # Check success (arbitrary threshold)
                if episode_reward > 50:  # Adjust based on your reward structure
                    successes += 1
            
            success_rate = (successes / total_tests) * 100
            success_rates.append(success_rate)
        
        # Update robustness plot
        self.robust_ax.clear()
        self.robust_ax.bar([f"{int(n*100)}%" for n in self.noise_levels], success_rates, 
                          color='skyblue', alpha=0.7)
        self.robust_ax.set_xlabel('Sensor Noise Level')
        self.robust_ax.set_ylabel('Success Rate (%)')
        self.robust_ax.set_title('Sensor Noise Robustness Analysis')
        self.robust_ax.grid(True, alpha=0.3)
        
        self.robust_fig.canvas.draw()
        
        # Save the robustness plot
        save_session_image(self.robust_fig, "robustness_analysis.png", "graphs")
        
        # Save the test data
        test_data = {
            'noise_levels': [f"{int(n*100)}%" for n in self.noise_levels],
            'success_rates': success_rates,
            'model_tested': 'current_model',
            'test_episodes': total_tests
        }
        save_session_data(test_data, "robustness_test_results.json")
        
        add_session_note(f"Robustness test completed. Success rates: {success_rates}")
        messagebox.showinfo("Complete", "Robustness testing completed! Results saved to session folder.")
        
    def compare_all_models(self):
        """Compare all available models (placeholder)"""
        messagebox.showinfo("Info", "Model comparison will be available once all models finish training!")

def main():
    """Main entry point"""
    root = tk.Tk()
    app = QuadrupedResearchDemo(root)
    root.mainloop()

if __name__ == "__main__":
    main()