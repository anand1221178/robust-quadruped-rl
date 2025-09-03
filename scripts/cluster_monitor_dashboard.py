#!/usr/bin/env python3
"""
Real-time Cluster Training Monitor Dashboard
Monitors W&B logs and cluster job status for parallel SR2L/DR training
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np
import requests
import threading
import time
import subprocess
import json
from datetime import datetime, timedelta
import os
from pathlib import Path

class ClusterMonitorDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Cluster Training Monitor - SR2L & DR Parallel Training")
        self.root.geometry("1600x1000")
        
        # Training tracking
        self.training_jobs = {
            'SR2L': {'status': 'Not Started', 'progress': 0, 'reward': [], 'loss': [], 'job_id': None},
            'DR': {'status': 'Not Started', 'progress': 0, 'reward': [], 'loss': [], 'job_id': None}
        }
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        
        self.setup_ui()
        self.start_monitoring()
        
    def setup_ui(self):
        """Setup the dashboard UI"""
        
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(title_frame, text="🚀 Cluster Training Monitor Dashboard", 
                 font=("Arial", 18, "bold")).pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(title_frame, text="⏸ Monitoring Paused", 
                                     font=("Arial", 12))
        self.status_label.pack(side=tk.RIGHT)
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Controls & Status
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel - Plots
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_control_panel(left_frame)
        self.setup_visualization_panel(right_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel"""
        
        # Job Control
        job_frame = ttk.LabelFrame(parent, text="Training Jobs Control")
        job_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(job_frame, text="🚀 Launch SR2L Training", 
                  command=self.launch_sr2l_training, 
                  style="Accent.TButton").pack(fill=tk.X, pady=2)
        
        ttk.Button(job_frame, text="🛡️ Launch DR Training", 
                  command=self.launch_dr_training,
                  style="Accent.TButton").pack(fill=tk.X, pady=2)
        
        ttk.Separator(job_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        ttk.Button(job_frame, text="📊 Check Job Status", 
                  command=self.check_job_status).pack(fill=tk.X, pady=2)
        
        ttk.Button(job_frame, text="🔄 Refresh Data", 
                  command=self.refresh_data).pack(fill=tk.X, pady=2)
        
        # Job Status Display
        status_frame = ttk.LabelFrame(parent, text="Current Job Status")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # SR2L Status
        sr2l_frame = ttk.LabelFrame(status_frame, text="SR2L Training")
        sr2l_frame.pack(fill=tk.X, pady=2)
        
        self.sr2l_status = ttk.Label(sr2l_frame, text="Status: Not Started")
        self.sr2l_status.pack(anchor=tk.W, padx=5)
        
        self.sr2l_progress = ttk.Progressbar(sr2l_frame, mode='determinate')
        self.sr2l_progress.pack(fill=tk.X, padx=5, pady=2)
        
        self.sr2l_info = tk.Text(sr2l_frame, height=4, width=35, font=('Consolas', 8))
        self.sr2l_info.pack(fill=tk.X, padx=5, pady=2)
        
        # DR Status
        dr_frame = ttk.LabelFrame(status_frame, text="DR Training")
        dr_frame.pack(fill=tk.X, pady=2)
        
        self.dr_status = ttk.Label(dr_frame, text="Status: Not Started")
        self.dr_status.pack(anchor=tk.W, padx=5)
        
        self.dr_progress = ttk.Progressbar(dr_frame, mode='determinate')
        self.dr_progress.pack(fill=tk.X, padx=5, pady=2)
        
        self.dr_info = tk.Text(dr_frame, height=4, width=35, font=('Consolas', 8))
        self.dr_info.pack(fill=tk.X, padx=5, pady=2)
        
        # System Info
        system_frame = ttk.LabelFrame(parent, text="System Info")
        system_frame.pack(fill=tk.X, pady=5)
        
        self.system_info = tk.Text(system_frame, height=6, width=35, font=('Consolas', 8))
        self.system_info.pack(fill=tk.X, padx=5, pady=2)
        
        self.update_system_info()
        
    def setup_visualization_panel(self, parent):
        """Setup the visualization panel"""
        
        # Create notebook for different plots
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Training Progress Tab
        progress_frame = ttk.Frame(notebook)
        notebook.add(progress_frame, text="📈 Training Progress")
        
        self.setup_progress_plots(progress_frame)
        
        # Performance Comparison Tab
        comparison_frame = ttk.Frame(notebook)
        notebook.add(comparison_frame, text="⚖️ Model Comparison")
        
        self.setup_comparison_plots(comparison_frame)
        
        # System Resources Tab
        resources_frame = ttk.Frame(notebook)
        notebook.add(resources_frame, text="💻 System Resources")
        
        self.setup_resource_plots(resources_frame)
        
    def setup_progress_plots(self, parent):
        """Setup training progress plots"""
        
        self.progress_fig, ((self.reward_ax, self.loss_ax), 
                           (self.lr_ax, self.fps_ax)) = plt.subplots(2, 2, figsize=(12, 8))
        self.progress_fig.suptitle('Training Progress - SR2L vs DR')
        
        # Reward plot
        self.reward_ax.set_title('Episode Reward')
        self.reward_ax.set_ylabel('Mean Reward')
        self.reward_ax.grid(True, alpha=0.3)
        self.reward_ax.legend(['SR2L', 'DR'])
        
        # Loss plot
        self.loss_ax.set_title('Policy Loss')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.grid(True, alpha=0.3)
        
        # Learning rate
        self.lr_ax.set_title('Learning Rate')
        self.lr_ax.set_ylabel('LR')
        self.lr_ax.grid(True, alpha=0.3)
        
        # FPS
        self.fps_ax.set_title('Training Speed')
        self.fps_ax.set_ylabel('FPS')
        self.fps_ax.set_xlabel('Training Steps')
        self.fps_ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(self.progress_fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_comparison_plots(self, parent):
        """Setup model comparison plots"""
        
        self.comp_fig, ((self.velocity_ax, self.smoothness_ax), 
                       (self.robustness_ax, self.radar_ax)) = plt.subplots(2, 2, figsize=(12, 8))
        self.comp_fig.suptitle('SR2L vs DR - Performance Comparison')
        
        # Velocity comparison
        self.velocity_ax.set_title('Walking Velocity')
        self.velocity_ax.set_ylabel('Velocity (m/s)')
        
        # Smoothness comparison  
        self.smoothness_ax.set_title('Action Smoothness')
        self.smoothness_ax.set_ylabel('Smoothness Score')
        
        # Robustness comparison
        self.robustness_ax.set_title('Noise Robustness')
        self.robustness_ax.set_ylabel('Success Rate (%)')
        
        # Radar chart
        self.radar_ax.set_title('Overall Performance Radar')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(self.comp_fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_resource_plots(self, parent):
        """Setup system resource monitoring"""
        
        self.resource_fig, ((self.cpu_ax, self.mem_ax), 
                          (self.gpu_ax, self.disk_ax)) = plt.subplots(2, 2, figsize=(12, 8))
        self.resource_fig.suptitle('Cluster Resource Usage')
        
        # CPU usage
        self.cpu_ax.set_title('CPU Usage')
        self.cpu_ax.set_ylabel('CPU %')
        self.cpu_ax.set_ylim(0, 100)
        
        # Memory usage
        self.mem_ax.set_title('Memory Usage')
        self.mem_ax.set_ylabel('Memory %')
        self.mem_ax.set_ylim(0, 100)
        
        # GPU usage
        self.gpu_ax.set_title('GPU Usage')
        self.gpu_ax.set_ylabel('GPU %')
        self.gpu_ax.set_ylim(0, 100)
        
        # Disk I/O
        self.disk_ax.set_title('Disk I/O')
        self.disk_ax.set_ylabel('MB/s')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(self.resource_fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def launch_sr2l_training(self):
        """Launch SR2L training on cluster"""
        try:
            cmd = ["sbatch", "scripts/train_ppo_cluster.sh", "ppo_sr2l_corrected"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                # Extract job ID from sbatch output
                job_id = result.stdout.strip().split()[-1]
                self.training_jobs['SR2L']['job_id'] = job_id
                self.training_jobs['SR2L']['status'] = f'Submitted (Job ID: {job_id})'
                
                self.sr2l_info.delete(1.0, tk.END)
                self.sr2l_info.insert(1.0, f"✅ Job submitted successfully!\\nJob ID: {job_id}\\nConfig: ppo_sr2l_corrected.yaml\\nExpected duration: ~24h")
                
                self.log_message(f"SR2L training launched with Job ID: {job_id}")
            else:
                self.log_message(f"Failed to launch SR2L: {result.stderr}")
                
        except Exception as e:
            self.log_message(f"Error launching SR2L: {str(e)}")
            
    def launch_dr_training(self):
        """Launch DR training on cluster"""
        try:
            cmd = ["sbatch", "scripts/train_ppo_cluster.sh", "ppo_dr_robust"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                # Extract job ID from sbatch output
                job_id = result.stdout.strip().split()[-1]
                self.training_jobs['DR']['job_id'] = job_id
                self.training_jobs['DR']['status'] = f'Submitted (Job ID: {job_id})'
                
                self.dr_info.delete(1.0, tk.END)
                self.dr_info.insert(1.0, f"✅ Job submitted successfully!\\nJob ID: {job_id}\\nConfig: ppo_dr_robust.yaml\\nExpected duration: ~24h")
                
                self.log_message(f"DR training launched with Job ID: {job_id}")
            else:
                self.log_message(f"Failed to launch DR: {result.stderr}")
                
        except Exception as e:
            self.log_message(f"Error launching DR: {str(e)}")
    
    def check_job_status(self):
        """Check status of submitted jobs"""
        for job_name, job_info in self.training_jobs.items():
            if job_info['job_id']:
                try:
                    cmd = ["squeue", "-j", job_info['job_id'], "--format=%T,%M,%R"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0 and len(result.stdout.strip().split('\\n')) > 1:
                        status_line = result.stdout.strip().split('\\n')[1]
                        state, time_used, reason = status_line.split(',')
                        
                        job_info['status'] = f'{state} ({time_used})'
                        
                        # Update UI
                        if job_name == 'SR2L':
                            self.sr2l_status.config(text=f"Status: {job_info['status']}")
                        else:
                            self.dr_status.config(text=f"Status: {job_info['status']}")
                            
                    else:
                        job_info['status'] = 'Unknown/Completed'
                        
                except Exception as e:
                    self.log_message(f"Error checking {job_name} status: {str(e)}")
    
    def refresh_data(self):
        """Refresh training data from logs/W&B"""
        self.log_message("Refreshing training data...")
        
        # Check for new log files
        self.scan_log_files()
        
        # Update plots
        self.update_progress_plots()
        
        # Check job status
        self.check_job_status()
        
    def scan_log_files(self):
        """Scan for new training log files"""
        log_patterns = {
            'SR2L': 'ppo_*sr2l*.out',
            'DR': 'ppo_*dr*.out'
        }
        
        for job_name, pattern in log_patterns.items():
            try:
                # Find matching log files
                import glob
                log_files = glob.glob(pattern)
                
                if log_files:
                    latest_log = max(log_files, key=os.path.getctime)
                    
                    # Parse training progress from log
                    with open(latest_log, 'r') as f:
                        lines = f.readlines()
                        
                    # Simple parsing for rewards/losses (adjust based on actual log format)
                    rewards = []
                    losses = []
                    
                    for line in lines:
                        if 'reward' in line.lower():
                            try:
                                reward = float(line.split(':')[-1].strip())
                                rewards.append(reward)
                            except:
                                continue
                                
                        if 'loss' in line.lower():
                            try:
                                loss = float(line.split(':')[-1].strip())
                                losses.append(loss)
                            except:
                                continue
                    
                    self.training_jobs[job_name]['reward'] = rewards[-50:]  # Keep last 50
                    self.training_jobs[job_name]['loss'] = losses[-50:]
                    
            except Exception as e:
                self.log_message(f"Error scanning logs for {job_name}: {str(e)}")
    
    def update_progress_plots(self):
        """Update the progress plots with latest data"""
        
        # Clear previous plots
        self.reward_ax.clear()
        self.loss_ax.clear()
        
        # Plot SR2L data
        if self.training_jobs['SR2L']['reward']:
            self.reward_ax.plot(self.training_jobs['SR2L']['reward'], 
                              label='SR2L', color='blue', linewidth=2)
        
        if self.training_jobs['SR2L']['loss']:
            self.loss_ax.plot(self.training_jobs['SR2L']['loss'], 
                            label='SR2L', color='blue', linewidth=2)
        
        # Plot DR data
        if self.training_jobs['DR']['reward']:
            self.reward_ax.plot(self.training_jobs['DR']['reward'], 
                              label='DR', color='red', linewidth=2)
        
        if self.training_jobs['DR']['loss']:
            self.loss_ax.plot(self.training_jobs['DR']['loss'], 
                            label='DR', color='red', linewidth=2)
        
        # Formatting
        self.reward_ax.set_title('Episode Reward')
        self.reward_ax.set_ylabel('Mean Reward')
        self.reward_ax.grid(True, alpha=0.3)
        self.reward_ax.legend()
        
        self.loss_ax.set_title('Policy Loss')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_ax.legend()
        
        # Refresh canvas
        self.progress_fig.canvas.draw()
    
    def start_monitoring(self):
        """Start the monitoring loop"""
        self.monitoring = True
        self.status_label.config(text="🔄 Monitoring Active")
        
        def monitor_loop():
            while self.monitoring:
                self.refresh_data()
                time.sleep(30)  # Refresh every 30 seconds
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring = False
        self.status_label.config(text="⏸ Monitoring Paused")
    
    def update_system_info(self):
        """Update system information display"""
        info_text = f"""System Status:
Time: {datetime.now().strftime('%H:%M:%S')}
Cluster: Active
Queue: bigbatch

Training Config:
- SR2L: Joint sensor perturbations
- DR: Progressive actuator failures  
- Duration: ~24h each
- Parallel execution: Enabled

Expected Results:
- Baseline: 0.216 m/s (smooth)
- SR2L: Sensor noise robustness
- DR: Actuator failure robustness"""
        
        self.system_info.delete(1.0, tk.END)
        self.system_info.insert(1.0, info_text)
    
    def log_message(self, message):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Configure ttk styles
    style = ttk.Style()
    style.theme_use('clam')
    
    app = ClusterMonitorDashboard(root)
    
    # Handle window closing
    def on_closing():
        app.stop_monitoring()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()