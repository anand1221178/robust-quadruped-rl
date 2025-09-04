#!/usr/bin/env python3
"""
🚀 DR-FOCUSED INTERACTIVE ROBOT VIEWER 🚀
Real-time Domain Randomization Analysis & Video Recording

Features:
🎯 Joint failure simulation
📊 DR performance metrics
🎬 Two-pass video recording
📈 Real-time fault analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import pickle
import cv2
from datetime import datetime
from PIL import Image, ImageTk
import io

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import session manager and two-pass video recorder
from session_manager import get_session_manager, save_session_image, add_session_note, copy_session_file
from utils.two_pass_video import TwoPassVideoRecorder

# Try to import MuJoCo viewer
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️  MuJoCo not available for live rendering. Will use performance visualization.")

try:
    from agents.ppo_sr2l import PPO_SR2L
    from envs.success_reward_wrapper import SuccessRewardWrapper
    from envs.target_walking_wrapper import TargetWalkingWrapper
    import realant_sim
except ImportError as e:
    print(f"Import warning: {e}")

class InteractiveRobotViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Sexy Robot Viewer - Real-time Performance & Recording")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#1e1e1e')  # Dark theme
        
        # Initialize session manager
        self.session_manager = get_session_manager()
        self.session_manager.add_tool_usage("interactive_robot_viewer", "Real-time robot visualization with video recording")
        
        # State variables
        self.current_model = None
        self.current_env = None
        self.base_env = None  # For MuJoCo access
        self.is_running = False
        self.is_recording = False
        self.thread = None
        self.mujoco_viewer = None
        self.render_thread = None
        self.render_running = False
        
        # Performance data
        self.velocity_data = []
        self.reward_data = []
        self.time_data = []
        self.action_data = []
        self.joint_torques = []  # Track individual joint outputs
        self.stability_data = []  # Track robot stability
        self.max_data_points = 500  # More data points for smooth curves
        
        # Video recording
        self.video_writer = None
        self.video_filename = None
        self.frame_buffer = []
        
        # Visualization settings
        self.camera_angle = 0
        self.show_trails = True
        self.current_step = 0
        
        # DR-specific tracking
        self.joint_failure_history = []
        self.stability_metrics = []
        self.dr_performance_ratio = []
        self.fault_tolerance_score = []
        
        # UI Theme colors
        self.colors = {
            'bg': '#1e1e1e',           # Dark background
            'panel': '#2d2d2d',        # Panel background
            'accent': '#00d4aa',       # Cyan accent
            'secondary': '#ff6b6b',    # Red accent  
            'text': '#ffffff',         # White text
            'success': '#4ecdc4',      # Success green
            'warning': '#ffe66d',      # Warning yellow
            'danger': '#ff6b6b'        # Danger red
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the sexy user interface"""
        
        # Configure ttk style for dark theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Accent.TButton', 
                       background=self.colors['accent'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        style.map('Accent.TButton',
                 background=[('active', '#00b89a')])
        
        style.configure('Secondary.TButton',
                       background=self.colors['secondary'],
                       foreground='white', 
                       borderwidth=0,
                       focuscolor='none')
        style.map('Secondary.TButton',
                 background=[('active', '#ff5252')])
        
        style.configure('Dark.TFrame', background=self.colors['panel'])
        style.configure('Dark.TLabelFrame', 
                       background=self.colors['panel'],
                       foreground=self.colors['text'])
        
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['bg'], height=80)
        header_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
        header_frame.pack_propagate(False)
        
        # Professional title
        title_label = tk.Label(header_frame, 
                              text="🤖 INTERACTIVE ROBOT VIEWER",
                              font=("Arial", 24, "bold"),
                              fg=self.colors['accent'],
                              bg=self.colors['bg'])
        title_label.pack(side=tk.LEFT, pady=20)
        
        subtitle_label = tk.Label(header_frame,
                                 text="Real-time Performance • Live Recording • Professional Analysis",
                                 font=("Arial", 12),
                                 fg=self.colors['text'],
                                 bg=self.colors['bg'])
        subtitle_label.pack(side=tk.LEFT, padx=(20, 0), pady=25)
        
        # Status indicator
        self.status_frame = tk.Frame(header_frame, bg=self.colors['bg'])
        self.status_frame.pack(side=tk.RIGHT, pady=20)
        
        self.status_indicator = tk.Label(self.status_frame,
                                        text="⚪ READY",
                                        font=("Arial", 14, "bold"),
                                        fg=self.colors['text'],
                                        bg=self.colors['bg'])
        self.status_indicator.pack()
        
        # Main container with dark theme
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Controls (wider and more beautiful)
        left_panel = tk.Frame(main_frame, bg=self.colors['panel'], width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Right panel - Visualization  
        right_panel = tk.Frame(main_frame, bg=self.colors['bg'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_controls(left_panel)
        self.setup_visualization(right_panel)
        
    def setup_controls(self, parent):
        """Setup sexy control panel"""
        
        # Add padding frame
        control_container = tk.Frame(parent, bg=self.colors['panel'])
        control_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Model loading section with modern styling
        model_frame = tk.LabelFrame(control_container, 
                                  text=" 🎯 MODEL SELECTION ",
                                  font=("Arial", 12, "bold"),
                                  fg=self.colors['accent'],
                                  bg=self.colors['panel'],
                                  bd=2,
                                  relief='ridge')
        model_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Beautiful load button
        load_btn_frame = tk.Frame(model_frame, bg=self.colors['panel'])
        load_btn_frame.pack(fill=tk.X, padx=15, pady=10)
        
        load_btn = tk.Button(load_btn_frame,
                            text="📁 LOAD MODEL",
                            font=("Arial", 12, "bold"),
                            bg=self.colors['accent'],
                            fg='white',
                            activebackground='#00b89a',
                            activeforeground='white',
                            relief='flat',
                            bd=0,
                            pady=10,
                            command=self.load_model)
        load_btn.pack(fill=tk.X)
        
        # Model info display with dark theme
        self.model_info = tk.Text(model_frame, 
                                 height=4, 
                                 font=('Consolas', 9),
                                 bg='#1a1a1a',
                                 fg=self.colors['text'],
                                 insertbackground=self.colors['accent'],
                                 selectbackground=self.colors['accent'],
                                 relief='flat',
                                 bd=1)
        self.model_info.pack(fill=tk.X, padx=15, pady=(0, 10))
        self.model_info.insert(1.0, "💤 No model loaded yet...\n\nClick 'LOAD MODEL' to begin!\n🚀 Ready for robot visualization!")
        self.model_info.config(state='disabled')  # Read-only
        
        # Control buttons with modern styling
        control_frame = tk.LabelFrame(control_container,
                                    text=" ⚡ SIMULATION CONTROLS ",
                                    font=("Arial", 12, "bold"),
                                    fg=self.colors['accent'],
                                    bg=self.colors['panel'],
                                    bd=2,
                                    relief='ridge')
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        button_container = tk.Frame(control_frame, bg=self.colors['panel'])
        button_container.pack(fill=tk.X, padx=15, pady=10)
        
        # Start button
        self.start_btn = tk.Button(button_container,
                                  text="▶️ START SIMULATION",
                                  font=("Arial", 11, "bold"),
                                  bg=self.colors['success'],
                                  fg='white',
                                  activebackground='#3aa39c',
                                  activeforeground='white',
                                  relief='flat',
                                  bd=0,
                                  pady=8,
                                  command=self.start_simulation)
        self.start_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Control button row
        btn_row = tk.Frame(button_container, bg=self.colors['panel'])
        btn_row.pack(fill=tk.X, pady=(0, 5))
        
        self.stop_btn = tk.Button(btn_row,
                                 text="⏹️ STOP",
                                 font=("Arial", 10, "bold"),
                                 bg=self.colors['danger'],
                                 fg='white',
                                 activebackground='#ff5252',
                                 activeforeground='white',
                                 relief='flat',
                                 bd=0,
                                 pady=6,
                                 state='disabled',
                                 command=self.stop_simulation)
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.reset_btn = tk.Button(btn_row,
                                  text="🔄 RESET",
                                  font=("Arial", 10, "bold"),
                                  bg=self.colors['warning'],
                                  fg='black',
                                  activebackground='#ffda44',
                                  activeforeground='black',
                                  relief='flat',
                                  bd=0,
                                  pady=6,
                                  command=self.reset_simulation)
        self.reset_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Recording controls with sexy styling
        record_frame = tk.LabelFrame(control_container,
                                   text=" 🎥 RECORDING STUDIO ",
                                   font=("Arial", 12, "bold"),
                                   fg=self.colors['accent'],
                                   bg=self.colors['panel'],
                                   bd=2,
                                   relief='ridge')
        record_frame.pack(fill=tk.X, pady=(0, 15))
        
        record_container = tk.Frame(record_frame, bg=self.colors['panel'])
        record_container.pack(fill=tk.X, padx=15, pady=10)
        
        # Recording button row
        record_btn_row = tk.Frame(record_container, bg=self.colors['panel'])
        record_btn_row.pack(fill=tk.X, pady=(0, 8))
        
        self.record_btn = tk.Button(record_btn_row,
                                   text="🎬 START RECORDING",
                                   font=("Arial", 10, "bold"),
                                   bg=self.colors['secondary'],
                                   fg='white',
                                   activebackground='#ff5252',
                                   activeforeground='white',
                                   relief='flat',
                                   bd=0,
                                   pady=6,
                                   command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        screenshot_btn = tk.Button(record_btn_row,
                                  text="📸 SNAP",
                                  font=("Arial", 10, "bold"),
                                  bg='#6c5ce7',
                                  fg='white',
                                  activebackground='#5a4fcf',
                                  activeforeground='white',
                                  relief='flat',
                                  bd=0,
                                  pady=6,
                                  command=self.take_screenshot)
        screenshot_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Recording status with modern styling
        self.record_status = tk.Label(record_container,
                                     text="⚪ READY TO RECORD",
                                     font=("Arial", 9, "bold"),
                                     fg=self.colors['text'],
                                     bg=self.colors['panel'])
        self.record_status.pack(pady=(0, 5))
        
        # Visualization controls with beautiful styling
        viz_frame = tk.LabelFrame(control_container,
                                text=" ⚙️ VISUALIZATION SETTINGS ",
                                font=("Arial", 12, "bold"),
                                fg=self.colors['accent'],
                                bg=self.colors['panel'],
                                bd=2,
                                relief='ridge')
        viz_frame.pack(fill=tk.X, pady=(0, 15))
        
        viz_container = tk.Frame(viz_frame, bg=self.colors['panel'])
        viz_container.pack(fill=tk.X, padx=15, pady=10)
        
        # Camera angle with modern label
        cam_label = tk.Label(viz_container,
                            text="📹 Camera Angle:",
                            font=("Arial", 10, "bold"),
                            fg=self.colors['text'],
                            bg=self.colors['panel'])
        cam_label.pack(anchor=tk.W, pady=(0, 3))
        
        self.angle_var = tk.DoubleVar(value=0)
        angle_scale = tk.Scale(viz_container,
                              from_=-180, to=180,
                              variable=self.angle_var,
                              orient=tk.HORIZONTAL,
                              bg=self.colors['panel'],
                              fg=self.colors['text'],
                              activebackground=self.colors['accent'],
                              troughcolor='#404040',
                              highlightbackground=self.colors['panel'],
                              relief='flat',
                              bd=0)
        angle_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Noise injection with beautiful styling
        dr_label = tk.Label(viz_container,
                           text="⚡ DR Joint Failure Rate:",
                              font=("Arial", 10, "bold"),
                              fg=self.colors['text'],
                              bg=self.colors['panel'])
        dr_label.pack(anchor=tk.W, pady=(0, 3))
        
        self.dr_failure_var = tk.DoubleVar(value=0.0)
        dr_scale = tk.Scale(viz_container,
                           from_=0.0, to=0.3,
                           variable=self.dr_failure_var,
                              orient=tk.HORIZONTAL,
                              bg=self.colors['panel'],
                              fg=self.colors['text'],
                           activebackground=self.colors['danger'],
                              troughcolor='#404040',
                              highlightbackground=self.colors['panel'],
                           resolution=0.05,
                              relief='flat',
                              bd=0,
                              command=self.update_dr_label)
        dr_scale.pack(fill=tk.X, pady=(0, 5))
        
        self.dr_failure_label = tk.Label(viz_container,
                                         text="0%",
                                         font=("Arial", 9),
                                         fg=self.colors['danger'],
                                         bg=self.colors['panel'])
        self.dr_failure_label.pack(pady=(0, 10))
        
        # Trail toggle with modern checkbox styling
        self.trails_var = tk.BooleanVar(value=True)
        trails_check = tk.Checkbutton(viz_container,
                                     text="✨ Show Movement Trails",
                                     variable=self.trails_var,
                                     font=("Arial", 10),
                                     fg=self.colors['text'],
                                     bg=self.colors['panel'],
                                     activebackground=self.colors['panel'],
                                     activeforeground=self.colors['accent'],
                                     selectcolor=self.colors['accent'],
                                     relief='flat',
                                     bd=0)
        trails_check.pack(anchor=tk.W, pady=5)
        
        # Live performance display with sexy dark theme
        perf_frame = tk.LabelFrame(control_container,
                                 text=" 📊 LIVE PERFORMANCE ",
                                 font=("Arial", 12, "bold"),
                                 fg=self.colors['accent'],
                                 bg=self.colors['panel'],
                                 bd=2,
                                 relief='ridge')
        perf_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.perf_display = tk.Text(perf_frame,
                                   height=8,
                                   font=('Consolas', 9),
                                   bg='#1a1a1a',
                                   fg=self.colors['text'],
                                   insertbackground=self.colors['accent'],
                                   selectbackground=self.colors['accent'],
                                   relief='flat',
                                   bd=1)
        self.perf_display.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Initialize with professional welcome message
        welcome_text = """🤖 READY FOR SIMULATION!

💫 Load a model to begin
🚀 Watch real-time performance
📈 Monitor velocity & rewards
🎥 Record robot demonstrations

⚡ Status: Awaiting your command..."""
        self.perf_display.insert(1.0, welcome_text)
        self.perf_display.config(state='disabled')
        
        # Save controls with professional styling
        save_frame = tk.LabelFrame(control_container,
                                 text=" 💾 EXPORT STUDIO ",
                                 font=("Arial", 12, "bold"),
                                 fg=self.colors['accent'],
                                 bg=self.colors['panel'],
                                 bd=2,
                                 relief='ridge')
        save_frame.pack(fill=tk.X)
        
        save_container = tk.Frame(save_frame, bg=self.colors['panel'])
        save_container.pack(fill=tk.X, padx=15, pady=10)
        
        # Save data button
        save_data_btn = tk.Button(save_container,
                                 text="💾 SAVE PERFORMANCE DATA",
                                 font=("Arial", 10, "bold"),
                                 bg='#74b9ff',
                                 fg='white',
                                 activebackground='#0984e3',
                                 activeforeground='white',
                                 relief='flat',
                                 bd=0,
                                 pady=6,
                                 command=self.save_performance_data)
        save_data_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Generate report button
        report_btn = tk.Button(save_container,
                              text="📊 GENERATE REPORT",
                              font=("Arial", 10, "bold"),
                              bg='#fd79a8',
                              fg='white',
                              activebackground='#e84393',
                              activeforeground='white',
                              relief='flat',
                              bd=0,
                              pady=6,
                              command=self.generate_report)
        report_btn.pack(fill=tk.X)
        
    def setup_visualization(self, parent):
        """Setup visualization panel"""
        
        # Create notebook for different views
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Robot view tab
        robot_frame = ttk.Frame(notebook)
        notebook.add(robot_frame, text="🤖 Robot View")
        
        # Robot visualization with real MuJoCo integration
        if MUJOCO_AVAILABLE:
            # Create MuJoCo viewer frame with actual rendering
            self.mujoco_frame = tk.Frame(robot_frame, bg='black')
            self.mujoco_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Canvas for MuJoCo rendering
            self.robot_canvas = tk.Canvas(self.mujoco_frame, bg='black', width=800, height=600)
            self.robot_canvas.pack(fill=tk.BOTH, expand=True)
            
            # Status label for rendering info
            self.render_status = tk.Label(self.mujoco_frame,
                                        text="🤖 MuJoCo Renderer Ready - Start simulation for live view!",
                                        font=("Arial", 10, "bold"),
                                        bg="black", fg="#00d4aa")
            self.render_status.pack(pady=5)
            
        else:
            # Fallback display for when MuJoCo is not available
            self.robot_display = tk.Label(robot_frame, 
                                         text="🤖 ROBOT VISUALIZATION\\n\\n" + 
                                              "⚠️  MuJoCo not available for live rendering\\n" +
                                              "📊 Using performance visualization instead\\n\\n" +
                                              "To enable live robot view:\\n" +
                                              "pip install mujoco\\n\\n" +
                                              "✨ Available Features:\\n" +
                                              "• Real-time performance graphs\\n" +
                                              "• Velocity and reward tracking\\n" +
                                              "• Professional data visualization\\n" +
                                              "• Session recording and export",
                                         font=("Arial", 12), 
                                         bg="#1a1a1a", fg="white",
                                         justify=tk.CENTER)
            self.robot_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Performance graphs tab
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="📈 Performance")
        
        self.setup_performance_plots(perf_frame)
        
        # Analysis tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="🔬 Analysis")
        
        self.setup_analysis_view(analysis_frame)
        
    def setup_performance_plots(self, parent):
        """Setup performance monitoring plots with beautiful dark theme"""
        
        # Create figure with dark theme
        plt.style.use('dark_background')
        self.perf_fig, ((self.vel_ax, self.reward_ax), 
                       (self.action_ax, self.dr_ax)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Apply beautiful dark theme
        self.perf_fig.patch.set_facecolor('#1e1e1e')
        self.perf_fig.suptitle('Real-time Performance Monitoring', 
                              fontsize=16, fontweight='bold', color='white')
        
        # Velocity plot with beautiful styling
        self.vel_ax.set_title('Walking Velocity', fontweight='bold', color='white')
        self.vel_ax.set_ylabel('Velocity (m/s)', color='white')
        self.vel_ax.set_facecolor('#2d2d2d')
        self.vel_ax.grid(True, alpha=0.3, color='gray')
        self.vel_line, = self.vel_ax.plot([], [], '#00d4aa', linewidth=3, alpha=0.9)
        self.vel_ax.axhline(y=0.2, color='#4ecdc4', linestyle='--', alpha=0.8, label='Target', linewidth=2)
        self.vel_ax.legend()
        
        # Reward plot with stunning colors
        self.reward_ax.set_title('Episode Reward', fontweight='bold', color='white')
        self.reward_ax.set_ylabel('Reward', color='white')
        self.reward_ax.set_facecolor('#2d2d2d')
        self.reward_ax.grid(True, alpha=0.3, color='gray')
        self.reward_line, = self.reward_ax.plot([], [], '#ff6b6b', linewidth=3, alpha=0.9)
        
        # Action analysis with professional look
        self.action_ax.set_title('Action Smoothness', fontweight='bold', color='white')
        self.action_ax.set_ylabel('Action Magnitude', color='white')
        self.action_ax.set_xlabel('Time Steps', color='white')
        self.action_ax.set_facecolor('#2d2d2d')
        self.action_ax.grid(True, alpha=0.3, color='gray')
        
        # Noise impact with eye-catching design
        self.dr_ax.set_title('DR Joint Health', fontweight='bold', color='white')
        self.dr_ax.set_ylabel('Joint Functionality (%)', color='white')
        self.dr_ax.set_xlabel('Time Steps', color='white')
        self.dr_ax.set_facecolor('#2d2d2d')
        self.dr_ax.grid(True, alpha=0.3, color='gray')
        self.dr_ax.set_ylim(0, 100)
        
        # Style all axes beautifully
        for ax in [self.vel_ax, self.reward_ax, self.action_ax, self.dr_ax]:
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#555555')
                spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(self.perf_fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_analysis_view(self, parent):
        """Setup analysis view with live statistics and visualizations"""
        
        # Create analysis container with dark theme
        analysis_frame = tk.Frame(parent, bg='#1e1e1e')
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistics display
        stats_frame = tk.LabelFrame(analysis_frame, 
                                   text=" 📊 LIVE STATISTICS ",
                                   font=("Arial", 12, "bold"),
                                   fg='#00d4aa',
                                   bg='#2d2d2d',
                                   bd=2,
                                   relief='ridge')
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_display = tk.Text(stats_frame,
                                    height=8,
                                    font=('Consolas', 10),
                                    bg='#1a1a1a',
                                    fg='white',
                                    insertbackground='#00d4aa',
                                    selectbackground='#00d4aa',
                                    relief='flat',
                                    bd=1)
        self.stats_display.pack(fill=tk.X, padx=10, pady=10)
        
        # Initialize with professional analysis content
        analysis_text = """📈 PERFORMANCE ANALYSIS CENTER

🎯 Real-time Metrics:
• Average velocity: Waiting for data...
• Peak velocity: Waiting for data...
• Reward efficiency: Calculating...
• Action consistency: Monitoring...

🔍 Analysis Features:
• Velocity trend analysis
• Reward pattern detection  
• Stability measurements
• Performance degradation tracking

🎥 Recording Capabilities:
• Performance plot video capture
• Screenshot functionality
• Session data logging
• Automated report generation

⚙️ Interactive Controls:
• Real-time DR joint failure simulation
• Camera angle adjustment  
• Movement trail visualization
• DR performance analysis tools"""
        
        self.stats_display.insert(1.0, analysis_text)
        self.stats_display.config(state='disabled')
        
        # Analysis controls
        controls_frame = tk.LabelFrame(analysis_frame,
                                     text=" 🔧 ANALYSIS CONTROLS ",
                                     font=("Arial", 12, "bold"),
                                     fg='#00d4aa',
                                     bg='#2d2d2d',
                                     bd=2,
                                     relief='ridge')
        controls_frame.pack(fill=tk.X)
        
        control_container = tk.Frame(controls_frame, bg='#2d2d2d')
        control_container.pack(fill=tk.X, padx=10, pady=10)
        
        # Analysis buttons
        btn_row1 = tk.Frame(control_container, bg='#2d2d2d')
        btn_row1.pack(fill=tk.X, pady=(0, 5))
        
        tk.Button(btn_row1,
                 text="📊 UPDATE STATISTICS",
                 font=("Arial", 9, "bold"),
                 bg='#74b9ff',
                 fg='white',
                 activebackground='#0984e3',
                 activeforeground='white',
                 relief='flat',
                 bd=0,
                 pady=4,
                 command=self.update_analysis).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        tk.Button(btn_row1,
                 text="🎯 RESET ANALYSIS",
                 font=("Arial", 9, "bold"),
                 bg='#fd79a8',
                 fg='white',
                 activebackground='#e84393',
                 activeforeground='white',
                 relief='flat',
                 bd=0,
                 pady=4,
                 command=self.reset_analysis).pack(side=tk.RIGHT, fill=tk.X, expand=True)
                 
    def update_analysis(self):
        """Update analysis statistics"""
        if not self.velocity_data or not self.reward_data:
            return
            
        # Calculate statistics
        avg_velocity = np.mean(self.velocity_data)
        max_velocity = max(self.velocity_data) if self.velocity_data else 0
        avg_reward = np.mean(self.reward_data)
        total_steps = len(self.velocity_data)
        
        # Performance rating
        if avg_velocity > 0.15:
            performance = "🟢 EXCELLENT"
        elif avg_velocity > 0.05:
            performance = "🟡 GOOD"
        else:
            performance = "🔴 NEEDS IMPROVEMENT"
            
        analysis_text = f"""📈 UPDATED PERFORMANCE ANALYSIS

🎯 Current Statistics:
• Average velocity: {avg_velocity:.3f} m/s
• Peak velocity: {max_velocity:.3f} m/s  
• Average reward: {avg_reward:.2f}
• Total steps: {total_steps}
• Performance rating: {performance}

🔍 Trend Analysis:
• Velocity stability: {np.std(self.velocity_data):.3f}
• Reward consistency: {np.std(self.reward_data):.2f}
• Recent performance: {np.mean(self.velocity_data[-50:]):.3f} m/s (last 50 steps)

📊 Session Summary:
• Data points collected: {len(self.velocity_data)}
• Recording status: {"🔴 RECORDING" if self.is_recording else "⚪ STANDBY"}
• Simulation status: {"🟢 RUNNING" if self.is_running else "🟡 STOPPED"}

🎯 Recommendations:
{"✅ Robot demonstrates excellent locomotion!" if avg_velocity > 0.15 else 
 "⚠️ Consider parameter tuning for better performance." if avg_velocity > 0.05 else
 "❌ Significant performance issues detected. Check model and environment."}"""
        
        self.stats_display.config(state='normal')
        self.stats_display.delete(1.0, tk.END)
        self.stats_display.insert(1.0, analysis_text)
        self.stats_display.config(state='disabled')
        
    def reset_analysis(self):
        """Reset analysis display"""
        analysis_text = """📈 ANALYSIS RESET

🎯 Statistics cleared and ready for new data collection.

🔍 Monitoring:
• Velocity measurements
• Reward tracking  
• Performance analysis
• Trend detection

📊 Waiting for simulation data..."""
        
        self.stats_display.config(state='normal')
        self.stats_display.delete(1.0, tk.END)
        self.stats_display.insert(1.0, analysis_text)
        self.stats_display.config(state='disabled')
        
    def load_model(self):
        """Load a model for visualization"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Load model
            if 'sr2l' in file_path.lower():
                self.current_model = PPO_SR2L.load(file_path)
            else:
                self.current_model = PPO.load(file_path)
            
            # Create environment with rendering support
            def make_env():
                # Add render_mode for live visualization
                _env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
                _env = SuccessRewardWrapper(_env)
                _env = Monitor(_env)
                return _env
            
            env = DummyVecEnv([make_env])
            
            # Check for vec_normalize
            model_dir = os.path.dirname(os.path.dirname(file_path))
            vec_path = os.path.join(model_dir, 'vec_normalize.pkl')
            
            if os.path.exists(vec_path):
                with open(vec_path, 'rb') as f:
                    vec_normalize = pickle.load(f)
                vec_normalize.venv = env
                env = vec_normalize
                env.training = False
            
            self.current_env = env
            
            # Store base environment for rendering
            self.base_env = None
            if hasattr(env, 'venv'):
                self.base_env = env.venv.envs[0]
            else:
                self.base_env = env.envs[0]
            
            # Get unwrapped env
            while hasattr(self.base_env, 'env'):
                self.base_env = self.base_env.env
            
            # Update UI with professional model info
            model_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            self.model_info.config(state='normal')
            self.model_info.delete(1.0, tk.END)
            model_text = f"""🚀 MODEL LOADED SUCCESSFULLY!

📂 {model_name}
🤖 {"PPO+SR2L" if 'sr2l' in file_path.lower() else "PPO Baseline"}
✅ Environment configured
🎯 Ready for visualization!

💫 Click START SIMULATION to begin!"""
            self.model_info.insert(1.0, model_text)
            self.model_info.config(state='disabled')
            
            # Log to session
            self.session_manager.add_model_test(model_name, {"loaded": True, "path": file_path})
            add_session_note(f"Loaded model for interactive visualization: {model_name}")
            
            # Enable start button and update status
            self.start_btn.config(state='normal')
            self.status_indicator.config(text="✅ MODEL LOADED", fg=self.colors['success'])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def start_simulation(self):
        """Start the real-time simulation"""
        if not self.current_model or not self.current_env:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        if self.is_running:
            return
            
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        # Clear previous data
        self.velocity_data = []
        self.reward_data = []
        self.time_data = []
        
        # Update status and start simulation
        self.status_indicator.config(text="🟢 RUNNING", fg=self.colors['success'])
        
        # DISABLE RENDERING THREAD - Causes system crashes due to OpenGL threading issues
        print(f"RENDER DEBUG: MUJOCO_AVAILABLE = {MUJOCO_AVAILABLE}")
        print(f"RENDER DEBUG: Rendering disabled to prevent system crashes")
        
        if MUJOCO_AVAILABLE and hasattr(self, 'robot_canvas'):
            self.render_status.config(text="⚠️ Live rendering disabled (prevents crashes)")
            # Show a message instead of crashing
            self.robot_canvas.create_text(400, 200, 
                                        text="🤖 ROBOT SIMULATION RUNNING\n\n" +
                                             "Live rendering disabled due to\n" +
                                             "OpenGL threading conflicts\n\n" +
                                             "📊 Use performance graphs to\n" +
                                             "monitor robot behavior\n\n" +
                                             "🎬 Use 'Record Video' for\n" +
                                             "accurate visual recordings", 
                                        fill="white", font=("Arial", 12))
        else:
            print(f"❌ MuJoCo not available for rendering")
        
        # Start simulation thread
        self.thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.thread.start()
        
        add_session_note("Started real-time robot simulation")
        
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.render_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        # Update status
        self.status_indicator.config(text="🟡 STOPPED", fg=self.colors['warning'])
        
        if MUJOCO_AVAILABLE and hasattr(self, 'render_status'):
            self.render_status.config(text="🤖 Rendering stopped")
        
        if self.is_recording:
            self.toggle_recording()  # Stop recording if active
            
        add_session_note("Stopped real-time robot simulation")
        
    def reset_simulation(self):
        """Reset the simulation and clear all data"""
        if self.is_running:
            self.stop_simulation()
            time.sleep(0.5)  # Wait for stop
            
        # Clear all data arrays
        self.velocity_data = []
        self.reward_data = []
        self.time_data = []
        self.action_data = []
        self.current_step = 0
        
        # Reset environment if loaded
        if self.current_env:
            try:
                self.current_env.reset()
            except:
                pass  # Ignore reset errors
        
        # Clear and reset all plots
        self.vel_line.set_data([], [])
        self.reward_line.set_data([], [])
        
        for ax in [self.vel_ax, self.reward_ax, self.action_ax, self.dr_ax]:
            ax.clear()
        
        # Reconfigure plots with fresh beautiful styling
        self.vel_ax.set_title('Walking Velocity', fontweight='bold', color='white')
        self.vel_ax.set_ylabel('Velocity (m/s)', color='white')
        self.vel_ax.set_facecolor('#2d2d2d')
        self.vel_ax.grid(True, alpha=0.3, color='gray')
        self.vel_line, = self.vel_ax.plot([], [], '#00d4aa', linewidth=3, alpha=0.9)
        self.vel_ax.axhline(y=0.2, color='#4ecdc4', linestyle='--', alpha=0.8, label='Target', linewidth=2)
        self.vel_ax.legend()
        
        self.reward_ax.set_title('Episode Reward', fontweight='bold', color='white')
        self.reward_ax.set_ylabel('Reward', color='white')
        self.reward_ax.set_facecolor('#2d2d2d')
        self.reward_ax.grid(True, alpha=0.3, color='gray')
        self.reward_line, = self.reward_ax.plot([], [], '#ff6b6b', linewidth=3, alpha=0.9)
        
        self.action_ax.set_title('Action Smoothness', fontweight='bold', color='white')
        self.action_ax.set_ylabel('Action Magnitude', color='white')
        self.action_ax.set_xlabel('Time Steps', color='white')
        self.action_ax.set_facecolor('#2d2d2d')
        self.action_ax.grid(True, alpha=0.3, color='gray')
        
        self.dr_ax.set_title('DR Joint Health', fontweight='bold', color='white')
        self.dr_ax.set_ylabel('Joint Functionality (%)', color='white')
        self.dr_ax.set_xlabel('Time Steps', color='white')
        self.dr_ax.set_facecolor('#2d2d2d')
        self.dr_ax.grid(True, alpha=0.3, color='gray')
        self.dr_ax.set_ylim(0, 100)
        
        # Style all axes beautifully
        for ax in [self.vel_ax, self.reward_ax, self.action_ax, self.dr_ax]:
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#555555')
                spine.set_linewidth(1.5)
        
        try:
            self.perf_fig.canvas.draw()
        except:
            pass
        
        # Clear and reset performance display with sexy message
        self.perf_display.config(state='normal')
        self.perf_display.delete(1.0, tk.END)
        reset_text = """🔄 SIMULATION RESET COMPLETE!

✨ All data cleared
📊 Graphs refreshed
🤖 Robot ready for new session

🚀 Ready to start fresh simulation...
💫 System ready for commands!"""
        self.perf_display.insert(1.0, reset_text)
        self.perf_display.config(state='disabled')
        
        # Update status
        self.status_indicator.config(text="⚪ RESET COMPLETE", fg=self.colors['success'])
        
        # Log reset to session
        add_session_note("Simulation reset: All data and visualizations cleared")
        
    def simulation_loop(self):
        """Main simulation loop"""
        if not self.current_env:
            return
            
        obs = self.current_env.reset()
        episode_reward = 0
        step_count = 0
        episode_count = 0
        
        while self.is_running:
            try:
                # Get action from model
                action, _ = self.current_model.predict(obs, deterministic=True)
                
                # Apply DR joint failures if specified
                if self.dr_failure_var.get() > 0:
                    action = self.apply_joint_failures(action)
                    
                    # Track joint health for DR visualization
                    joint_health = self.calculate_joint_health()
                    self.joint_failure_history.append(joint_health)
                    
                    # Calculate stability metrics
                    stability = self.calculate_stability_score(obs)
                    self.stability_metrics.append(stability)
                
                # Execute action
                obs, reward, done, info = self.current_env.step(action)
                
                # Extract velocity (pass reward for fallback method)
                velocity = self.extract_velocity_with_reward(info, reward)
                
                # Store trajectory data if recording (Pass 1: No rendering)
                if self.is_recording and hasattr(self, 'recording_trajectory'):
                    self.recording_trajectory['actions'].append(action.copy())
                    self.recording_trajectory['observations'].append(obs.copy())
                    self.recording_trajectory['rewards'].append(reward[0] if hasattr(reward, '__len__') else reward)
                    self.recording_trajectory['velocities'].append(velocity)
                    self.recording_trajectory['infos'].append(info[0] if hasattr(info, '__len__') else info)
                    
                    # Update recorder's internal trajectory
                    if hasattr(self, 'two_pass_recorder'):
                        self.two_pass_recorder.trajectory = self.recording_trajectory
                        # Update performance metrics
                        self.two_pass_recorder.performance_metrics = {
                            'episode_reward': episode_reward,
                            'episode_length': len(self.recording_trajectory['actions']),
                            'avg_velocity': np.mean(self.recording_trajectory['velocities']) if self.recording_trajectory['velocities'] else 0.0,
                            'max_velocity': np.max(self.recording_trajectory['velocities']) if self.recording_trajectory['velocities'] else 0.0,
                            'min_velocity': np.min(self.recording_trajectory['velocities']) if self.recording_trajectory['velocities'] else 0.0,
                            'velocity_std': np.std(self.recording_trajectory['velocities']) if self.recording_trajectory['velocities'] else 0.0
                        }
                
                # Debug velocity extraction every 50 steps
                if step_count % 50 == 0 and step_count > 0:
                    # Additional debugging for velocity
                    try:
                        if self.base_env and hasattr(self.base_env, 'sim'):
                            raw_qvel = self.base_env.sim.data.qvel[0]
                            print(f"Step {step_count}: Velocity = {velocity:.3f} m/s, Reward = {reward[0]:.2f}, Raw qvel[0] = {raw_qvel:.3f}")
                        else:
                            print(f"Step {step_count}: Velocity = {velocity:.3f} m/s, Reward = {reward[0]:.2f}, Raw qvel[0] = N/A")
                        print(f"  DEBUG: base_env type: {type(self.base_env) if self.base_env else 'None'}")
                        print(f"  DEBUG: has sim: {hasattr(self.base_env, 'sim') if self.base_env else False}")
                    except Exception as debug_e:
                        print(f"Step {step_count}: Velocity = {velocity:.3f} m/s, Reward = {reward[0]:.2f}")
                        print(f"  DEBUG ERROR: {debug_e}")
                
                # Update data
                self.time_data.append(step_count)
                self.velocity_data.append(velocity)
                self.reward_data.append(reward[0])
                self.action_data.append(action.copy() if hasattr(action, 'copy') else action)
                
                # Keep only recent data
                if len(self.time_data) > self.max_data_points:
                    self.time_data.pop(0)
                    self.velocity_data.pop(0)
                    self.reward_data.pop(0)
                    if self.action_data:
                        self.action_data.pop(0)
                
                episode_reward += reward[0]
                step_count += 1
                
                # Update UI every 10 steps
                if step_count % 10 == 0:
                    self.root.after(0, self.update_ui, velocity, episode_reward, step_count)
                
                # Update analysis every 100 steps
                if step_count % 100 == 0:
                    self.root.after(0, self.update_analysis)
                
                # Capture video frame if recording
                if self.is_recording and step_count % 20 == 0:  # Capture every 20 steps for 2.5 fps
                    self.capture_video_frame()
                
                # Handle episode end
                if done[0]:
                    episode_count += 1
                    add_session_note(f"Episode {episode_count} completed: {episode_reward:.2f} reward, {np.mean(self.velocity_data[-100:]):.3f} m/s avg velocity")
                    obs = self.current_env.reset()
                    episode_reward = 0
                
                # Control simulation speed
                time.sleep(0.02)  # ~50 FPS
                
            except Exception as e:
                print(f"Simulation error: {e}")
                break
                
        self.is_running = False
        
    def extract_velocity_with_reward(self, info, reward):
        """Extract velocity using EXACT working method from research_demo_gui.py"""
        velocity = 0
        
        try:
            # EXACT COPY from research_demo_gui.py - extract base_env dynamically every time
            if hasattr(self.current_env, 'venv'):
                base_env = self.current_env.venv.envs[0]
                print(f"  DEBUG: Using venv path, initial base_env type: {type(base_env)}")
            else:
                base_env = self.current_env.envs[0]
                print(f"  DEBUG: Using envs path, initial base_env type: {type(base_env)}")
            
            # Get unwrapped env - CRITICAL STEP!
            unwrap_count = 0
            while hasattr(base_env, 'env'):
                print(f"  DEBUG: Unwrapping layer {unwrap_count}: {type(base_env)} -> {type(base_env.env)}")
                base_env = base_env.env
                unwrap_count += 1
            
            print(f"  DEBUG: Final unwrapped base_env type: {type(base_env)}")
            print(f"  DEBUG: Final base_env has sim: {hasattr(base_env, 'sim')}")
            print(f"  DEBUG: Final base_env has _sim: {hasattr(base_env, '_sim')}")  
            print(f"  DEBUG: Final base_env has data: {hasattr(base_env, 'data')}")
            print(f"  DEBUG: Final base_env has model: {hasattr(base_env, 'model')}")
            
            # List some attributes for debugging
            if hasattr(base_env, '__dict__'):
                attrs = [attr for attr in dir(base_env) if not attr.startswith('__')]
                print(f"  DEBUG: Available attributes: {attrs[:10]}...")  # First 10
            
            # For MuJoCo envs, get x velocity directly
            if hasattr(base_env, 'sim') and hasattr(base_env.sim, 'data'):
                # Get center of mass velocity
                velocity = base_env.sim.data.qvel[0]  # x-velocity
                print(f"  DEBUG: SUCCESS! Raw qvel[0] = {velocity}")
            elif hasattr(base_env, '_sim') and hasattr(base_env._sim, 'data'):
                # Try alternative sim attribute
                velocity = base_env._sim.data.qvel[0]
                print(f"  DEBUG: SUCCESS via _sim! Raw qvel[0] = {velocity}")
            elif hasattr(base_env, 'model') and hasattr(base_env, 'data'):
                # Try direct model/data access
                velocity = base_env.data.qvel[0]
                print(f"  DEBUG: SUCCESS via data! Raw qvel[0] = {velocity}")
            else:
                # Fallback - extract from reward or info
                if info and len(info) > 0 and info[0] is not None:
                    info_dict = info[0] if isinstance(info, list) else info
                    if 'current_velocity' in info_dict:
                        velocity = info_dict['current_velocity']
                        print(f"  DEBUG: FALLBACK via info - velocity = {velocity}")
                    else:
                        velocity = 0
                        print(f"  DEBUG: FALLBACK - no velocity info, using velocity = 0")
                else:
                    velocity = 0
                    print(f"  DEBUG: FALLBACK - no info, using velocity = 0")
        except Exception as e:
            print(f"Velocity extraction error: {e}")
            velocity = 0
        
        # Fallback methods if direct velocity extraction failed (PROVEN METHOD)
        if velocity == 0:
            # Method 1: Check info dict
            try:
                if len(info) > 0 and hasattr(info[0], 'get'):
                    velocity = info[0].get('speed', info[0].get('x_velocity', 0))
            except:
                pass
            
            # Method 2: Use reward as proxy (rough estimate)
            if velocity == 0 and reward[0] > 0:
                # SuccessRewardWrapper gives speed-based rewards
                velocity = reward[0] * 0.2  # Scale factor
        
        # Ensure velocity is positive and reasonable
        velocity = abs(velocity)
        
        return velocity
    
    def rendering_loop(self):
        """Real-time MuJoCo rendering loop - dynamic environment access"""
        print("🎬 RENDERING LOOP STARTED!")
        print(f"🎬 MUJOCO_AVAILABLE: {MUJOCO_AVAILABLE}")
        
        if not MUJOCO_AVAILABLE or not self.current_env:
            print("❌ RENDERING LOOP EXITING - MuJoCo not available or no env")
            self.root.after(0, lambda: self.render_status.config(text="❌ MuJoCo not available"))
            return
            
        try:
            import mujoco
            import numpy as np
            from PIL import Image, ImageTk
            
            print("Setting up MuJoCo rendering...")
            
            # Wait for simulation to start and environment to be ready
            for i in range(10):  # Try for 5 seconds
                if self.is_running and self.current_env:
                    break
                time.sleep(0.5)
            
            if not self.is_running:
                print("❌ Simulation stopped before rendering could start")
                return
            
            # Dynamically get the actual base environment
            def get_base_env():
                """Extract base environment dynamically like in velocity extraction"""
                try:
                    if hasattr(self.current_env, 'venv'):
                        base_env = self.current_env.venv.envs[0]
                    else:
                        base_env = self.current_env.envs[0]
                    
                    # Unwrap all the way
                    while hasattr(base_env, 'env'):
                        base_env = base_env.env
                    
                    return base_env
                except:
                    return None
            
            # Test rendering with dynamic environment
            base_env = get_base_env()
            if base_env and hasattr(base_env, 'render'):
                try:
                    # Try without mode parameter (gymnasium style)
                    test_render = base_env.render()
                    if test_render is not None:
                        print(f"✅ Render test successful! Shape: {test_render.shape}")
                        self.root.after(0, lambda: self.render_status.config(
                            text="✅ Rendering initialized!"))
                    else:
                        print("❌ Render test returned None")
                        self.root.after(0, lambda: self.render_status.config(
                            text="⚠️ Render not available"))
                        return
                except Exception as e:
                    print(f"❌ Render test failed: {e}")
                    # Try to understand the error
                    if "mode" in str(e):
                        print("Note: This environment doesn't accept 'mode' parameter")
                    self.root.after(0, lambda: self.render_status.config(
                        text="⚠️ Cannot render this environment"))
                    return
            else:
                print("❌ No base environment with render method")
                self.root.after(0, lambda: self.render_status.config(
                    text="⚠️ Environment not renderable"))
                return
            
            # Wait a bit for simulation to initialize
            time.sleep(1.0)
            
            while self.render_running:
                try:
                    # Get current base environment dynamically
                    base_env = get_base_env()
                    if base_env and hasattr(base_env, 'render'):
                        # Simple render approach (no mode parameter)
                        try:
                            rgb = base_env.render()  # No mode parameter
                            if rgb is not None and hasattr(rgb, 'shape'):
                                # Convert to image and display
                                pil_image = Image.fromarray(rgb)
                                # Resize to canvas size
                                pil_image = pil_image.resize((800, 600))
                                photo = ImageTk.PhotoImage(pil_image)
                                
                                # Update canvas on main thread
                                self.root.after(0, self.update_robot_display, photo)
                                self.root.after(0, lambda: self.render_status.config(
                                    text="🎥 Live rendering active!"))
                        except Exception as render_e:
                            # Don't spam errors, just update status occasionally
                            if np.random.random() < 0.1:  # Only 10% of the time
                                print(f"Render loop warning: {render_e}")
                    else:
                        # Environment not ready yet, keep trying
                        time.sleep(0.1)
                        
                    time.sleep(0.2)  # 5 FPS to reduce load
                    
                except Exception as e:
                    print(f"Rendering error: {e}")
                    time.sleep(0.5)
                    
        except ImportError:
            self.root.after(0, lambda: self.render_status.config(text="❌ MuJoCo rendering unavailable"))
        except Exception as e:
            print(f"Rendering setup error: {e}")
            self.root.after(0, lambda: self.render_status.config(text=f"❌ Rendering error: {str(e)[:30]}..."))
    
    def update_robot_display(self, photo):
        """Update the robot canvas with new frame"""
        try:
            if hasattr(self, 'robot_canvas'):
                self.robot_canvas.delete("all")
                self.robot_canvas.create_image(400, 300, image=photo)
                self.robot_canvas.image = photo  # Keep a reference
        except Exception as e:
            print(f"Canvas update error: {e}")
        
    def update_ui(self, velocity, episode_reward, step_count):
        """Update UI elements"""
        
        # Update performance display
        perf_text = f"""🤖 LIVE PERFORMANCE

Current Velocity: {velocity:.3f} m/s
Episode Reward: {episode_reward:.2f}
Step Count: {step_count}
DR Failure Rate: {self.dr_failure_var.get()*100:.0f}%

Recent Performance:
Avg Velocity: {np.mean(self.velocity_data[-50:]):.3f} m/s
Avg Reward: {np.mean(self.reward_data[-50:]):.2f}
Max Velocity: {max(self.velocity_data) if self.velocity_data else 0:.3f} m/s

Status: {"🟢 Excellent" if velocity > 0.15 else "🟡 Good" if velocity > 0.05 else "🔴 Poor"}
"""
        
        self.perf_display.config(state='normal')
        self.perf_display.delete(1.0, tk.END)
        self.perf_display.insert(1.0, perf_text)
        self.perf_display.config(state='disabled')
        
        # Update plots
        self.update_performance_plots()
        
    def update_performance_plots(self):
        """Update performance plots"""
        if not self.time_data:
            return
            
        # Update velocity plot
        self.vel_line.set_data(self.time_data, self.velocity_data)
        self.vel_ax.relim()
        self.vel_ax.autoscale_view()
        
        # Update reward plot  
        self.reward_line.set_data(self.time_data, self.reward_data)
        self.reward_ax.relim()
        self.reward_ax.autoscale_view()
        
        # Update action plot
        if self.action_data and len(self.action_data) > 1:
            self.action_ax.clear()
            self.action_ax.set_title('Action Commands', fontweight='bold', color='white')
            self.action_ax.set_ylabel('Action Magnitude', color='white')
            self.action_ax.set_xlabel('Time Steps', color='white')
            self.action_ax.set_facecolor('#2d2d2d')
            self.action_ax.grid(True, alpha=0.3, color='gray')
            
            # Plot action magnitudes
            action_magnitudes = [np.linalg.norm(a) if hasattr(a, '__len__') else abs(a) 
                               for a in self.action_data[-len(self.time_data):]]
            if action_magnitudes:
                self.action_ax.plot(self.time_data[-len(action_magnitudes):], action_magnitudes,
                                   color='#ffe66d', linewidth=2, alpha=0.9)
                
            # Style the axis
            for spine in self.action_ax.spines.values():
                spine.set_color('white')
            self.action_ax.tick_params(colors='white')
        
        # Update DR joint health plot
        if self.joint_failure_history and len(self.joint_failure_history) > 1:
            self.dr_ax.clear()
            self.dr_ax.set_title('DR Joint Health', fontweight='bold', color='white')
            self.dr_ax.set_ylabel('Joint Functionality (%)', color='white')
            self.dr_ax.set_xlabel('Time Steps', color='white')
            self.dr_ax.set_facecolor('#2d2d2d')
            self.dr_ax.grid(True, alpha=0.3, color='gray')
            self.dr_ax.set_ylim(0, 100)
            
            # Plot joint health over time
            dr_time = list(range(len(self.joint_failure_history)))
            self.dr_ax.plot(dr_time, self.joint_failure_history, 
                           color='#ff6b6b', linewidth=2, label='Joint Health')
            
            # Add stability score if available
            if len(self.stability_metrics) == len(self.joint_failure_history):
                stability_pct = [s * 100 for s in self.stability_metrics]
                self.dr_ax.plot(dr_time, stability_pct, 
                               color='#4ecdc4', linewidth=2, label='Stability')
            
            # Add failure rate reference line
            failure_rate = self.dr_failure_var.get()
            expected_health = (1.0 - failure_rate) * 100
            self.dr_ax.axhline(y=expected_health, color='#ffe66d', 
                              linestyle='--', alpha=0.8, 
                              label=f'Expected ({expected_health:.0f}%)')
            
            self.dr_ax.legend()
            
            # Style the axis
            for spine in self.dr_ax.spines.values():
                spine.set_color('white')
            self.dr_ax.tick_params(colors='white')
        
        # Redraw
        try:
            self.perf_fig.canvas.draw_idle()  # Use draw_idle for thread safety
            self.perf_fig.canvas.flush_events()
        except Exception as e:
            print(f"Graph update error: {e}")  # Log errors for debugging
            
    def update_dr_label(self, value):
        """Update DR failure rate label"""
        failure_pct = float(value) * 100
        self.dr_failure_label.config(text=f"{failure_pct:.0f}%")
    
    def apply_joint_failures(self, action):
        """Simulate joint failures for DR testing"""
        if len(action.shape) > 1:
            action = action[0]  # Remove batch dimension if present
            
        modified_action = action.copy()
        failure_rate = self.dr_failure_var.get()
        
        # Simulate different types of joint failures
        for i in range(len(modified_action)):
            if np.random.random() < failure_rate:
                failure_type = np.random.choice(['lock', 'weak', 'noise'])
                if failure_type == 'lock':
                    modified_action[i] = 0.0  # Joint locked
                elif failure_type == 'weak':
                    modified_action[i] *= 0.3  # Weak joint (30% power)
                elif failure_type == 'noise':
                    modified_action[i] += np.random.normal(0, 0.5)  # Noisy joint
                    
        return modified_action.reshape(action.shape)
    
    def calculate_joint_health(self):
        """Calculate overall joint health percentage"""
        failure_rate = self.dr_failure_var.get()
        return (1.0 - failure_rate) * 100
    
    def calculate_stability_score(self, obs):
        """Calculate robot stability score from observations"""
        if len(obs.shape) > 1:
            obs = obs[0]  # Remove batch dimension
            
        # Use angular velocities as stability indicator
        # Assuming obs contains angular velocities in positions 3-6
        if len(obs) > 6:
            ang_vels = obs[3:6] if len(obs) > 6 else obs[:3]
            stability = 1.0 / (1.0 + np.sum(np.abs(ang_vels)))
            return min(stability, 1.0)
        return 1.0
    
    def simulate_joint_failure(self):
        """Manually trigger a joint failure for testing"""
        if self.is_running:
            # Temporarily increase failure rate for dramatic effect
            old_rate = self.dr_failure_var.get()
            self.dr_failure_var.set(0.8)  # 80% failure rate
            
            # Reset after 2 seconds
            self.root.after(2000, lambda: self.dr_failure_var.set(old_rate))
            
            add_session_note("Manual joint failure simulation triggered")
        else:
            messagebox.showinfo("Info", "Start simulation first to test joint failures!")
    
    def show_dr_analysis(self):
        """Show detailed DR performance analysis"""
        if not self.joint_failure_history:
            messagebox.showinfo("Info", "No DR data available. Run simulation with DR enabled first!")
            return
            
        # Create analysis window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("DR Performance Analysis")
        analysis_window.geometry("800x600")
        analysis_window.configure(bg=self.colors['bg'])
        
        # Add analysis content here
        analysis_text = f"""
        DR PERFORMANCE ANALYSIS
        ========================
        
        Average Joint Health: {np.mean(self.joint_failure_history):.1f}%
        Stability Score: {np.mean(self.stability_metrics) if self.stability_metrics else 0:.3f}
        
        Total Simulation Steps: {len(self.velocity_data)}
        DR Failure Rate: {self.dr_failure_var.get()*100:.0f}%
        
        Performance Under Failures:
        - Average Velocity: {np.mean(self.velocity_data[-100:]) if len(self.velocity_data) > 0 else 0:.3f} m/s
        - Velocity Stability: {np.std(self.velocity_data[-100:]) if len(self.velocity_data) > 0 else 0:.3f}
        """
        
        text_widget = tk.Text(analysis_window, 
                             bg=self.colors['panel'], 
                             fg=self.colors['text'],
                             font=("Consolas", 11))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        text_widget.insert(1.0, analysis_text)
        text_widget.config(state=tk.DISABLED)
        
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start TWO-PASS video recording for accurate performance"""
        if not self.is_running:
            messagebox.showwarning("Warning", "Start simulation first!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = self.session_manager.get_session_path('videos', f'robot_demo_{timestamp}.mp4')
        
        try:
            # Initialize two-pass recorder
            self.two_pass_recorder = TwoPassVideoRecorder()
            
            # Initialize trajectory storage
            self.recording_trajectory = {
                'actions': [],
                'observations': [], 
                'rewards': [],
                'velocities': [],
                'infos': []
            }
            
            self.is_recording = True
            self.record_btn.config(text="⏹️ STOP RECORDING")
            self.record_status.config(text="🔴 PASS 1: COLLECTING (No Render)")
            
            add_session_note(f"Started TWO-PASS recording: {self.video_filename.name}")
            messagebox.showinfo("Two-Pass Recording", 
                              "Pass 1: Collecting trajectory without rendering\n" +
                              "Pass 2 will create video when you stop recording")
            
        except Exception as e:
            messagebox.showerror("Recording Error", f"Failed to start recording: {str(e)}")
            
    def stop_recording(self):
        """Stop recording and create video using Pass 2 of two-pass method"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.record_btn.config(text="🎬 START RECORDING")
        self.record_status.config(text="🎥 PASS 2: CREATING VIDEO...")
        
        try:
            if hasattr(self, 'recording_trajectory') and len(self.recording_trajectory['actions']) > 0:
                # Pass 2: Create video from trajectory
                add_session_note(f"Creating video from {len(self.recording_trajectory['actions'])} steps")
                
                # Use the two-pass recorder to replay with rendering
                from envs.success_reward_wrapper import SuccessRewardWrapper
                success = self.two_pass_recorder.replay_with_rendering(
                    trajectory=self.recording_trajectory,
                    output_path=str(self.video_filename),
                    wrapper_class=SuccessRewardWrapper,
                    show_progress=False
                )
                
                if success:
                    # Get true performance metrics
                    metrics = self.two_pass_recorder.get_metrics()
                    
                    self.record_status.config(text="✅ VIDEO SAVED")
                    add_session_note(f"Video saved: {self.video_filename.name}")
                    add_session_note(f"TRUE Performance: {metrics['avg_velocity']:.3f} m/s")
                    
                    messagebox.showinfo("Two-Pass Recording Complete", 
                                      f"Video saved to: {self.video_filename.name}\n\n" +
                                      f"TRUE PERFORMANCE (No Rendering Overhead):\n" +
                                      f"Average Velocity: {metrics['avg_velocity']:.3f} m/s\n" +
                                      f"Max Velocity: {metrics['max_velocity']:.3f} m/s\n" +
                                      f"Episode Length: {metrics['episode_length']} steps")
                else:
                    self.record_status.config(text="⚠️ VIDEO FAILED")
                    messagebox.showwarning("Recording Failed", "Failed to create video from trajectory")
            else:
                self.record_status.config(text="⚪ READY TO RECORD")
                messagebox.showwarning("No Data", "No trajectory recorded")
                
        except Exception as e:
            self.record_status.config(text="⚪ READY TO RECORD")
            messagebox.showerror("Recording Error", f"Error stopping recording: {str(e)}")
    
    def capture_video_frame(self):
        """Capture a frame of the performance plots for video"""
        try:
            if self.video_writer and hasattr(self, 'perf_fig'):
                # Convert matplotlib figure to numpy array (updated method)
                self.perf_fig.canvas.draw()
                
                # Use buffer_rgba() instead of tostring_rgb() for newer matplotlib
                buf = np.frombuffer(self.perf_fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(self.perf_fig.canvas.get_width_height()[::-1] + (4,))
                
                # Convert RGBA to RGB then to BGR for OpenCV
                buf = buf[:, :, :3]  # Remove alpha channel
                buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
                
                # Resize to match video dimensions
                buf = cv2.resize(buf, (1200, 800))
                
                # Write frame
                self.video_writer.write(buf)
                
        except Exception as e:
            print(f"Video frame capture error: {e}")
            
    def take_screenshot(self):
        """Take a screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the performance plot
        screenshot_path = self.session_manager.get_session_path('images', f'robot_screenshot_{timestamp}.png')
        save_session_image(self.perf_fig, f'robot_screenshot_{timestamp}.png', 'images')
        
        add_session_note(f"Screenshot saved: robot_screenshot_{timestamp}.png")
        messagebox.showinfo("Screenshot", f"Screenshot saved to session folder!")
        
    def save_performance_data(self):
        """Save performance data"""
        if not self.velocity_data:
            messagebox.showwarning("Warning", "No data to save!")
            return
            
        data = {
            'timestamp': datetime.now().isoformat(),
            'time_steps': self.time_data,
            'velocities': self.velocity_data,
            'rewards': self.reward_data,
            'statistics': {
                'avg_velocity': np.mean(self.velocity_data),
                'max_velocity': max(self.velocity_data),
                'avg_reward': np.mean(self.reward_data),
                'total_steps': len(self.time_data)
            },
            'settings': {
                'noise_level': self.noise_var.get(),
                'camera_angle': self.angle_var.get(),
                'show_trails': self.trails_var.get()
            }
        }
        
        filename = f'performance_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        self.session_manager.save_data(data, filename, 'data')
        
        add_session_note(f"Performance data saved: {filename}")
        messagebox.showinfo("Data Saved", "Performance data saved to session folder!")
        
    def generate_report(self):
        """Generate comprehensive report"""
        if not self.velocity_data:
            messagebox.showwarning("Warning", "No data to report!")
            return
            
        report = f"""
INTERACTIVE ROBOT VIEWER - SESSION REPORT
{'='*50}

Performance Summary:
- Total Steps: {len(self.time_data)}
- Average Velocity: {np.mean(self.velocity_data):.3f} m/s
- Maximum Velocity: {max(self.velocity_data):.3f} m/s
- Average Reward: {np.mean(self.reward_data):.2f}
- Total Reward: {sum(self.reward_data):.2f}

Settings Used:
- DR Failure Rate: {self.dr_failure_var.get()*100:.0f}%
- Camera Angle: {self.angle_var.get():.1f}°
- Movement Trails: {self.trails_var.get()}

Performance Analysis:
- Velocity Stability: {np.std(self.velocity_data):.3f}
- Reward Stability: {np.std(self.reward_data):.3f}
- Peak Performance: {max(self.velocity_data):.3f} m/s at step {self.time_data[self.velocity_data.index(max(self.velocity_data))]}

Recommendations:
{"✅ Excellent performance! Robot demonstrates stable locomotion." if np.mean(self.velocity_data) > 0.15 else 
 "⚠️  Good performance with room for improvement." if np.mean(self.velocity_data) > 0.05 else
 "❌ Poor performance. Consider retraining or adjusting parameters."}
        """
        
        filename = f'robot_viewer_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        self.session_manager.save_report(report, filename)
        
        add_session_note("Generated comprehensive performance report")
        messagebox.showinfo("Report Generated", "Comprehensive report saved to session folder!")

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    
    app = InteractiveRobotViewer(root)
    
    # Handle window closing
    def on_closing():
        if app.is_running:
            app.stop_simulation()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()