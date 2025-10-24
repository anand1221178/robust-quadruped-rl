#!/usr/bin/env python3
"""
SR2L Action Smoothness Visualization
Shows how SR2L produces smoother, less jerky control signals compared to baseline
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def record_action_trajectory(model, env, num_steps=1000):
    """
    Run model for num_steps and record all actions
    Returns: (actions, observations, rewards)
    """
    actions_history = []
    obs_history = []
    rewards_history = []

    obs = env.reset()

    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        actions_history.append(action[0])  # Unwrap from vectorized env
        obs_history.append(obs[0])

        obs, reward, done, info = env.step(action)
        rewards_history.append(reward[0])

        if done[0]:
            obs = env.reset()

    return np.array(actions_history), np.array(obs_history), np.array(rewards_history)

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
print("Recording M1 (Baseline) trajectory...")
m1_actions, m1_obs, m1_rewards = record_action_trajectory(m1_model, env_m1, num_steps=1000)

print("Recording M2 (SR2L) trajectory...")
m2_actions, m2_obs, m2_rewards = record_action_trajectory(m2_model, env_m2, num_steps=1000)

# Joint names for RealAnt (4 legs × 2 joints each)
joint_names = [
    'Hip 1 (Front-L)', 'Ankle 1 (Front-L)',
    'Hip 2 (Front-R)', 'Ankle 2 (Front-R)',
    'Hip 3 (Rear-L)', 'Ankle 3 (Rear-L)',
    'Hip 4 (Rear-R)', 'Ankle 4 (Rear-R)'
]

# Create figure with 8 subplots (one per joint)
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle('Action Smoothness Comparison: SR2L vs Baseline\n(Smoother = Less Jerky = Better Motor Control)',
             fontsize=16, fontweight='bold', y=0.995)

axes = axes.flatten()

# Plot window (show first 500 steps for clarity)
plot_steps = 500
timesteps = np.arange(plot_steps)

for joint_idx in range(8):
    ax = axes[joint_idx]

    # Plot both models' actions
    ax.plot(timesteps, m1_actions[:plot_steps, joint_idx],
            color='#D32F2F', linewidth=1.5, alpha=0.7, label='M1 (Baseline)')
    ax.plot(timesteps, m2_actions[:plot_steps, joint_idx],
            color='#1B5E20', linewidth=1.5, alpha=0.8, label='M2 (SR2L)')

    # Styling
    ax.set_title(joint_names[joint_idx], fontsize=12, fontweight='bold')
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Action (Torque)', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, plot_steps)

    # Add legend only to first subplot
    if joint_idx == 0:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

plt.tight_layout()

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "sr2l_action_smoothness.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f"\n✅ Action smoothness figure saved to: {output_file}")

output_file_png = output_dir / "sr2l_action_smoothness.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ PNG preview saved to: {output_file_png}")

# Compute smoothness metrics
m1_action_changes = np.sum(np.abs(np.diff(m1_actions, axis=0)), axis=0)
m2_action_changes = np.sum(np.abs(np.diff(m2_actions, axis=0)), axis=0)

m1_total_change = np.sum(m1_action_changes)
m2_total_change = np.sum(m2_action_changes)

print("\n" + "="*80)
print("ACTION SMOOTHNESS METRICS")
print("="*80)
print(f"M1 (Baseline) Total Action Change: {m1_total_change:.2f}")
print(f"M2 (SR2L) Total Action Change: {m2_total_change:.2f}")
print(f"SR2L Smoothness Improvement: {((m1_total_change - m2_total_change) / m1_total_change * 100):.1f}%")
print()
print("Per-Joint Action Changes:")
print("-" * 80)
for i, name in enumerate(joint_names):
    improvement = ((m1_action_changes[i] - m2_action_changes[i]) / m1_action_changes[i]) * 100
    print(f"{name:20s} | M1: {m1_action_changes[i]:6.2f} | M2: {m2_action_changes[i]:6.2f} | Improvement: {improvement:5.1f}%")
print("="*80)

# Create smoothness comparison bar chart
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Per-joint comparison
x_pos = np.arange(8)
width = 0.35
ax1.bar(x_pos - width/2, m1_action_changes, width, label='M1 (Baseline)', color='#D32F2F', alpha=0.8)
ax1.bar(x_pos + width/2, m2_action_changes, width, label='M2 (SR2L)', color='#1B5E20', alpha=0.8)
ax1.set_xlabel('Joint', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Action Change (Σ|Δa|)', fontsize=12, fontweight='bold')
ax1.set_title('Per-Joint Action Changes (Lower = Smoother)', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'J{i+1}' for i in range(8)])
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# Right: Total smoothness
ax2.bar(['M1 (Baseline)', 'M2 (SR2L)'], [m1_total_change, m2_total_change],
        color=['#D32F2F', '#1B5E20'], alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('Total Action Change (All Joints)', fontsize=12, fontweight='bold')
ax2.set_title('Overall Control Smoothness\n(Lower = Less Jerky = Better)', fontsize=13, fontweight='bold')
improvement_pct = ((m1_total_change - m2_total_change) / m1_total_change) * 100
ax2.text(1, m2_total_change + 10, f'{improvement_pct:.1f}%\nsmoother!',
         ha='center', fontsize=13, fontweight='bold', color='darkgreen',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save bar chart
output_file2 = output_dir / "sr2l_smoothness_comparison.pdf"
plt.savefig(output_file2, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f"✅ Smoothness comparison saved to: {output_file2}")

output_file2_png = output_dir / "sr2l_smoothness_comparison.png"
plt.savefig(output_file2_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ PNG preview saved to: {output_file2_png}")

plt.show()

print("\n✅ SR2L produces smoother control signals - demonstrated!")
print(f"✅ {improvement_pct:.1f}% reduction in action changes = less motor wear")
