#!/usr/bin/env python3
"""
Demo Tools Launcher
Easy access to all research demonstration tools
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
from pathlib import Path

class DemoLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Robust Quadruped RL - Demo Tools Launcher")
        self.root.geometry("600x500")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the launcher UI"""
        
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(title_frame, text="🚀 Demo Tools Launcher", 
                 font=("Arial", 20, "bold")).pack()
        
        ttk.Label(title_frame, text="Robust Quadruped RL Research Tools", 
                 font=("Arial", 12), foreground="gray").pack()
        
        # Tools section
        tools_frame = ttk.LabelFrame(self.root, text="Available Tools")
        tools_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Tool buttons
        tools = [
            {
                'name': '🤖 Interactive Robot Viewer',
                'description': 'Real-time robot visualization with video recording',
                'script': 'interactive_robot_viewer.py',
                'color': '#E91E63'
            },
            {
                'name': '🎮 Interactive Research Demo',
                'description': 'Interactive GUI for testing models with live visualization',
                'script': 'research_demo_gui.py',
                'color': '#4CAF50'
            },
            {
                'name': '📊 Cluster Training Monitor',
                'description': 'Real-time monitoring dashboard for parallel training jobs',
                'script': 'cluster_monitor_dashboard.py',
                'color': '#2196F3'
            },
            {
                'name': '📈 Ablation Study Visualizer',
                'description': 'Comprehensive 4-way comparison visualization',
                'script': 'ablation_study_visualizer.py',
                'color': '#FF9800'
            },
            {
                'name': '🛡️ Robustness Evaluation Suite',
                'description': 'Comprehensive failure mode testing and analysis',
                'script': 'comprehensive_robustness_suite.py',
                'color': '#9C27B0'
            }
        ]
        
        for i, tool in enumerate(tools):
            tool_frame = ttk.Frame(tools_frame)
            tool_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Tool button
            btn = ttk.Button(tool_frame, text=tool['name'], 
                           command=lambda s=tool['script']: self.launch_tool(s),
                           width=30)
            btn.pack(side=tk.LEFT)
            
            # Description
            ttk.Label(tool_frame, text=tool['description'], 
                     foreground="gray", font=("Arial", 9)).pack(side=tk.LEFT, padx=(10, 0))
        
        # Status section
        status_frame = ttk.LabelFrame(self.root, text="Current Status")
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        status_text = """Project Status:
✅ Baseline Model: Available (done/ppo_baseline_ueqbjf2x)  
🔄 SR2L Training: In Progress (ppo_sr2l_corrected)
🔄 DR Training: In Progress (ppo_dr_robust)
⏳ Combined Model: Planned (after individual completion)

Quick Commands:
• Launch SR2L: sbatch scripts/train_ppo_cluster.sh ppo_sr2l_corrected
• Launch DR: sbatch scripts/train_ppo_cluster.sh ppo_dr_robust
• Check Status: squeue -u $USER"""
        
        status_label = ttk.Label(status_frame, text=status_text, 
                               font=("Consolas", 9), justify=tk.LEFT)
        status_label.pack(padx=10, pady=10)
        
        # Control buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(control_frame, text="📁 Open Scripts Folder", 
                  command=self.open_scripts_folder).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="📖 View Documentation", 
                  command=self.view_documentation).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="❓ Help", 
                  command=self.show_help).pack(side=tk.RIGHT)
        
    def launch_tool(self, script_name):
        """Launch a specific tool"""
        script_path = os.path.join('scripts', script_name)
        
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found: {script_path}")
            return
        
        try:
            # Launch in a new process
            subprocess.Popen([sys.executable, script_path], 
                           cwd=os.path.dirname(os.path.dirname(__file__)))
            
            messagebox.showinfo("Success", f"Launched {script_name}!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch {script_name}:\\n{str(e)}")
    
    def open_scripts_folder(self):
        """Open scripts folder in file manager"""
        scripts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
        
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", scripts_path])
            elif sys.platform == "win32":  # Windows
                subprocess.run(["explorer", scripts_path])
            else:  # Linux
                subprocess.run(["xdg-open", scripts_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")
    
    def view_documentation(self):
        """View project documentation"""
        claude_md_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLAUDE.md')
        
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", claude_md_path])
            elif sys.platform == "win32":  # Windows
                subprocess.run(["notepad", claude_md_path])
            else:  # Linux
                subprocess.run(["xdg-open", claude_md_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open documentation: {str(e)}")
    
    def show_help(self):
        """Show help information"""
        help_text = """Robust Quadruped RL - Demo Tools Help

🎮 Interactive Research Demo:
   • Load and test different models interactively
   • Real-time performance visualization
   • Noise injection and robustness testing
   • Live action smoothness monitoring

📊 Cluster Training Monitor:
   • Monitor parallel training jobs on cluster
   • Launch new training jobs with sbatch
   • Real-time progress visualization
   • Job status and resource monitoring

📈 Ablation Study Visualizer:
   • Comprehensive 4-way comparison charts
   • 11 different visualization types
   • Performance radar charts
   • Research timeline and status

🛡️ Robustness Evaluation Suite:
   • Test 6 different failure modes
   • Automated statistical analysis
   • Comprehensive reporting
   • Heatmaps and degradation analysis

Commands:
   python scripts/research_demo_gui.py
   python scripts/cluster_monitor_dashboard.py
   python scripts/ablation_study_visualizer.py --save results
   python scripts/comprehensive_robustness_suite.py --episodes 50

For more help, check CLAUDE.md or the project documentation.
"""
        
        # Create help window
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - Demo Tools")
        help_window.geometry("700x500")
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(1.0, help_text)
        text_widget.config(state=tk.DISABLED)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(help_window, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    
    app = DemoLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()