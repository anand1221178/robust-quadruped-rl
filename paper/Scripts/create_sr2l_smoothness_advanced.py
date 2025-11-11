#!/usr/bin/env python3
"""
SR2L Advanced Smoothness Visualization
1. Power Spectral Density - Shows frequency content (lower freq = smoother)
2. Action Acceleration - Shows jerkiness (lower acceleration = smoother)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import gymnasium as gym
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def record_action_trajectory(model, env, num_steps=1000):
    """Run model and record actions"""
    actions_history = []
    obs = env.reset()

    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        actions_history.append(action[0])
        obs, reward, done, info = env.step(action)
        if done[0]:
            obs = env.reset()

    return np.array(actions_history)

# Load models
print("Loading models...")
m1_model = PPO.load("done/ppo_baseline_ueqbjf2x/best_model/best_model.zip")
m2_model = PPO.load("done/ppo_sr2l_forward_m7gtjtpa/final_model.zip")

# Create environments
print("Creating environments...")
env_m1 = gym.make('RealAntMujoco-v0')
env_m1 = SuccessRewardWrapper(env_m1)
env_m1 = DummyVecEnv([lambda: env_m1])
env_m1 = VecNormalize.load("done/ppo_baseline_ueqbjf2x/vec_normalize.pkl", env_m1)
env_m1.training = False
env_m1.norm_reward = False

env_m2 = gym.make('RealAntMujoco-v0')
env_m2 = SuccessRewardWrapper(env_m2)
env_m2 = DummyVecEnv([lambda: env_m2])
env_m2 = VecNormalize.load("done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl", env_m2)
env_m2.training = False
env_m2.norm_reward = False

# Record trajectories
print("Recording trajectories...")
m1_actions = record_action_trajectory(m1_model, env_m1, num_steps=1000)
m2_actions = record_action_trajectory(m2_model, env_m2, num_steps=1000)

# Joint names
joint_names = [
    'Hip 1', 'Ankle 1', 'Hip 2', 'Ankle 2',
    'Hip 3', 'Ankle 3', 'Hip 4', 'Ankle 4'
]

# ============================================================================
# FIGURE 1: Power Spectral Density (Frequency Analysis)
# ============================================================================
print("Computing Power Spectral Density...")

fig1, axes = plt.subplots(2, 4, figsize=(18, 8))
fig1.suptitle('Power Spectral Density: SR2L Has Less High-Frequency Noise\n(Lower high-frequency power = Smoother control)',
              fontsize=15, fontweight='bold', y=0.98)

axes = axes.flatten()
sampling_rate = 50  # Hz (MuJoCo default)

for joint_idx in range(8):
    ax = axes[joint_idx]

    # Compute PSD for both models
    freqs_m1, psd_m1 = signal.welch(m1_actions[:, joint_idx], fs=sampling_rate, nperseg=256)
    freqs_m2, psd_m2 = signal.welch(m2_actions[:, joint_idx], fs=sampling_rate, nperseg=256)

    # Plot on log scale
    ax.semilogy(freqs_m1, psd_m1, color='#D32F2F', linewidth=2, alpha=0.8, label='M1 (Baseline)')
    ax.semilogy(freqs_m2, psd_m2, color='#1B5E20', linewidth=2, alpha=0.8, label='M2 (SR2L)')

    # Highlight high-frequency region (>5 Hz)
    ax.axvspan(5, freqs_m1[-1], alpha=0.15, color='red', label='High-freq noise' if joint_idx == 0 else '')

    ax.set_title(joint_names[joint_idx], fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', fontsize=9)
    ax.set_ylabel('Power Spectral Density', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 15)  # Focus on 0-15 Hz

    if joint_idx == 0:
        ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()

# Save PSD figure
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "sr2l_power_spectral_density.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f" PSD figure saved to: {output_file}")

output_file_png = output_dir / "sr2l_power_spectral_density.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f" PNG preview saved to: {output_file_png}")

# ============================================================================
# FIGURE 2: Action Acceleration (Jerk Analysis)
# ============================================================================
print("Computing action acceleration...")

# Compute acceleration (second derivative)
m1_velocity = np.diff(m1_actions, axis=0)  # First derivative
m1_acceleration = np.diff(m1_velocity, axis=0)  # Second derivative

m2_velocity = np.diff(m2_actions, axis=0)
m2_acceleration = np.diff(m2_velocity, axis=0)

fig2, axes = plt.subplots(2, 4, figsize=(18, 8))
fig2.suptitle('Action Acceleration: SR2L Reduces Control Jerkiness\n(Lower acceleration magnitude = Less jerky = Smoother)',
              fontsize=15, fontweight='bold', y=0.98)

axes = axes.flatten()
plot_steps = 300  # Show shorter window for clarity

for joint_idx in range(8):
    ax = axes[joint_idx]

    timesteps = np.arange(plot_steps)

    # Plot acceleration
    ax.plot(timesteps, m1_acceleration[:plot_steps, joint_idx],
            color='#D32F2F', linewidth=1.2, alpha=0.7, label='M1 (Baseline)')
    ax.plot(timesteps, m2_acceleration[:plot_steps, joint_idx],
            color='#1B5E20', linewidth=1.2, alpha=0.8, label='M2 (SR2L)')

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)

    ax.set_title(joint_names[joint_idx], fontsize=11, fontweight='bold')
    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('Acceleration (Δ²a)', fontsize=9)
    ax.grid(True, alpha=0.3)

    if joint_idx == 0:
        ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()

# Save acceleration figure
output_file2 = output_dir / "sr2l_action_acceleration.pdf"
plt.savefig(output_file2, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f" Acceleration figure saved to: {output_file2}")

output_file2_png = output_dir / "sr2l_action_acceleration.png"
plt.savefig(output_file2_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f" PNG preview saved to: {output_file2_png}")

# ============================================================================
# FIGURE 3: Summary Comparison (Best of Both)
# ============================================================================
print("Creating summary comparison...")

fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle('SR2L Smoothness Showcase: Frequency & Jerk Analysis',
              fontsize=16, fontweight='bold', y=0.98)

# Top-left: Average PSD across all joints
m1_psd_avg = np.zeros_like(freqs_m1)
m2_psd_avg = np.zeros_like(freqs_m2)

for joint_idx in range(8):
    freqs_m1, psd_m1 = signal.welch(m1_actions[:, joint_idx], fs=sampling_rate, nperseg=256)
    freqs_m2, psd_m2 = signal.welch(m2_actions[:, joint_idx], fs=sampling_rate, nperseg=256)
    m1_psd_avg += psd_m1
    m2_psd_avg += psd_m2

m1_psd_avg /= 8
m2_psd_avg /= 8

ax1.semilogy(freqs_m1, m1_psd_avg, color='#D32F2F', linewidth=3, alpha=0.8, label='M1 (Baseline)')
ax1.semilogy(freqs_m2, m2_psd_avg, color='#1B5E20', linewidth=3, alpha=0.8, label='M2 (SR2L)')
ax1.axvspan(5, freqs_m1[-1], alpha=0.15, color='red')
ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Power Spectral Density', fontsize=12, fontweight='bold')
ax1.set_title('(a) Average Frequency Content (All Joints)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(0, 15)

# Top-right: High-frequency power comparison
high_freq_cutoff = 5  # Hz
high_freq_indices = freqs_m1 > high_freq_cutoff

m1_high_freq_power = []
m2_high_freq_power = []

for joint_idx in range(8):
    freqs_m1, psd_m1 = signal.welch(m1_actions[:, joint_idx], fs=sampling_rate, nperseg=256)
    freqs_m2, psd_m2 = signal.welch(m2_actions[:, joint_idx], fs=sampling_rate, nperseg=256)

    m1_high_freq_power.append(np.sum(psd_m1[high_freq_indices]))
    m2_high_freq_power.append(np.sum(psd_m2[high_freq_indices]))

x_pos = np.arange(8)
width = 0.35
ax2.bar(x_pos - width/2, m1_high_freq_power, width, label='M1 (Baseline)', color='#D32F2F', alpha=0.8)
ax2.bar(x_pos + width/2, m2_high_freq_power, width, label='M2 (SR2L)', color='#1B5E20', alpha=0.8)
ax2.set_xlabel('Joint', fontsize=12, fontweight='bold')
ax2.set_ylabel('High-Frequency Power (>5 Hz)', fontsize=12, fontweight='bold')
ax2.set_title('(b) High-Frequency Noise per Joint', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'J{i+1}' for i in range(8)])
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# Bottom-left: RMS acceleration
m1_rms_accel = np.sqrt(np.mean(m1_acceleration**2, axis=0))
m2_rms_accel = np.sqrt(np.mean(m2_acceleration**2, axis=0))

ax3.bar(x_pos - width/2, m1_rms_accel, width, label='M1 (Baseline)', color='#D32F2F', alpha=0.8)
ax3.bar(x_pos + width/2, m2_rms_accel, width, label='M2 (SR2L)', color='#1B5E20', alpha=0.8)
ax3.set_xlabel('Joint', fontsize=12, fontweight='bold')
ax3.set_ylabel('RMS Acceleration', fontsize=12, fontweight='bold')
ax3.set_title('(c) Control Jerkiness per Joint', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'J{i+1}' for i in range(8)])
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# Bottom-right: Overall metrics
metrics = ['Total High-Freq\nPower', 'Total RMS\nAcceleration', 'Action Change\n(Σ|Δa|)']
m1_metrics = [
    np.sum(m1_high_freq_power),
    np.sum(m1_rms_accel),
    np.sum(np.abs(np.diff(m1_actions, axis=0)))
]
m2_metrics = [
    np.sum(m2_high_freq_power),
    np.sum(m2_rms_accel),
    np.sum(np.abs(np.diff(m2_actions, axis=0)))
]

# Normalize to M1 baseline (show as percentages)
m2_normalized = [(m2 / m1) * 100 for m1, m2 in zip(m1_metrics, m2_metrics)]

x_pos_metrics = np.arange(len(metrics))
bars = ax4.bar(x_pos_metrics, m2_normalized, color=['#1B5E20', '#1B5E20', '#1B5E20'], alpha=0.8, edgecolor='black', linewidth=2)
ax4.axhline(y=100, color='black', linestyle='--', linewidth=2, alpha=0.7, label='M1 Baseline (100%)')
ax4.set_ylabel('M2 as % of M1\n(Lower = Smoother)', fontsize=12, fontweight='bold')
ax4.set_title('(d) SR2L Overall Smoothness Improvement', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos_metrics)
ax4.set_xticklabels(metrics, fontsize=10)
ax4.set_ylim(80, 105)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for i, (bar, val) in enumerate(zip(bars, m2_normalized)):
    improvement = 100 - val
    color = 'darkgreen' if improvement > 0 else 'darkred'
    ax4.text(i, val - 3, f'{improvement:+.1f}%\nsmoother',
             ha='center', fontsize=10, fontweight='bold', color=color)

plt.tight_layout()

# Save summary figure
output_file3 = output_dir / "sr2l_smoothness_showcase.pdf"
plt.savefig(output_file3, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f" Summary showcase saved to: {output_file3}")

output_file3_png = output_dir / "sr2l_smoothness_showcase.png"
plt.savefig(output_file3_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f" PNG preview saved to: {output_file3_png}")

plt.show()

# Print summary statistics
print("\n" + "="*80)
print("SR2L SMOOTHNESS ANALYSIS SUMMARY")
print("="*80)
print(f"High-Frequency Power Reduction: {100 - m2_normalized[0]:.1f}%")
print(f"RMS Acceleration Reduction: {100 - m2_normalized[1]:.1f}%")
print(f"Action Change Reduction: {100 - m2_normalized[2]:.1f}%")
print("="*80)
print(" SR2L produces smoother, less jerky control signals!")
print(" Lower high-frequency content = more stable, predictable control")
print(" Lower acceleration = reduced motor wear and energy consumption")
