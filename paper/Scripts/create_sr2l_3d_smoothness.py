#!/usr/bin/env python3
"""
SR2L 3D Showcase: Policy Action Smoothness Under Noise
Visualizes how SR2L maintains consistent actions despite observation perturbations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def compute_action_variance(model, env, base_obs, noise_levels, num_samples=100):
    """
    For a given observation, add noise and measure action variance
    Returns: (noise_levels, mean_action_variance)
    """
    variances = []

    for noise_level in noise_levels:
        actions_collected = []

        for _ in range(num_samples):
            # Add noise to joints only (dims 13-28)
            noisy_obs = base_obs.copy()
            joint_noise = np.random.normal(0, noise_level, 16)
            noisy_obs[0, 13:29] += joint_noise

            # Get action
            action, _ = model.predict(noisy_obs, deterministic=True)
            actions_collected.append(action[0])

        # Compute variance across all actions for this noise level
        actions_array = np.array(actions_collected)  # (num_samples, 8)
        variance = np.mean(np.var(actions_array, axis=0))  # Mean variance across all joints
        variances.append(variance)

    return noise_levels, variances

def sample_diverse_observations(env, num_states=50):
    """Sample diverse states from environment"""
    observations = []

    for _ in range(num_states):
        obs = env.reset()
        # Run for random number of steps to get diversity
        for _ in range(np.random.randint(0, 200)):
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            if done[0]:
                obs = env.reset()
                break
        observations.append(obs)

    return observations

# Load models
print("Loading models...")
m1_model = PPO.load("done/ppo_baseline_ueqbjf2x/best_model/best_model.zip")
m2_model = PPO.load("done/ppo_sr2l_forward_m7gtjtpa/final_model.zip")

# Create environments
env_m1 = gym.make('RealAntMujoco-v0')
env_m1 = SuccessRewardWrapper(env_m1)
env_m1 = DummyVecEnv([lambda: env_m1])
env_m1 = VecNormalize.load("done/ppo_baseline_ueqbjf2x/vec_normalize.pkl", env_m1)
env_m1.training = False

env_m2 = gym.make('RealAntMujoco-v0')
env_m2 = SuccessRewardWrapper(env_m2)
env_m2 = DummyVecEnv([lambda: env_m2])
env_m2 = VecNormalize.load("done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl", env_m2)
env_m2.training = False

# Sample observations
print("Sampling diverse observations...")
observations = sample_diverse_observations(env_m1, num_states=20)

# Noise levels to test
noise_levels = np.linspace(0.0, 0.1, 15)

# Compute action variance for both models across all sampled states
print("Computing action smoothness...")
m1_variances_all = []
m2_variances_all = []

for i, obs in enumerate(observations):
    print(f"  State {i+1}/{len(observations)}...")
    _, m1_vars = compute_action_variance(m1_model, env_m1, obs, noise_levels, num_samples=50)
    _, m2_vars = compute_action_variance(m2_model, env_m2, obs, noise_levels, num_samples=50)
    m1_variances_all.append(m1_vars)
    m2_variances_all.append(m2_vars)

m1_variances_all = np.array(m1_variances_all)  # (num_states, num_noise_levels)
m2_variances_all = np.array(m2_variances_all)

# Create 3D visualization
fig = plt.figure(figsize=(16, 6))

# ===== LEFT: M1 (Baseline) Action Variance Surface =====
ax1 = fig.add_subplot(121, projection='3d')

X, Y = np.meshgrid(range(len(observations)), noise_levels)
Z1 = m1_variances_all.T  # Transpose to match meshgrid shape

surf1 = ax1.plot_surface(X, Y, Z1, cmap='Reds', alpha=0.8, edgecolor='none')
ax1.set_xlabel('State Index', fontsize=11, fontweight='bold')
ax1.set_ylabel('Noise Level (σ)', fontsize=11, fontweight='bold')
ax1.set_zlabel('Action Variance', fontsize=11, fontweight='bold')
ax1.set_title('M1 (Baseline): Action Sensitivity to Noise\n(Higher variance = less smooth)',
              fontsize=13, fontweight='bold', pad=20)
ax1.view_init(elev=20, azim=45)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Variance')

# ===== RIGHT: M2 (SR2L) Action Variance Surface =====
ax2 = fig.add_subplot(122, projection='3d')

Z2 = m2_variances_all.T

surf2 = ax2.plot_surface(X, Y, Z2, cmap='Greens', alpha=0.8, edgecolor='none')
ax2.set_xlabel('State Index', fontsize=11, fontweight='bold')
ax2.set_ylabel('Noise Level (σ)', fontsize=11, fontweight='bold')
ax2.set_zlabel('Action Variance', fontsize=11, fontweight='bold')
ax2.set_title('M2 (SR2L): Action Smoothness Under Noise\n(Lower variance = smoother policy)',
              fontsize=13, fontweight='bold', pad=20, color='darkgreen')
ax2.view_init(elev=20, azim=45)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Variance')

plt.tight_layout()

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "sr2l_3d_smoothness.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f"\n 3D smoothness figure saved to: {output_file}")

output_file_png = output_dir / "sr2l_3d_smoothness.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f" PNG preview saved to: {output_file_png}")

plt.show()

# Summary statistics
print("\n" + "="*80)
print("SR2L SMOOTHNESS ANALYSIS")
print("="*80)
m1_mean_var = np.mean(m1_variances_all)
m2_mean_var = np.mean(m2_variances_all)
smoothness_improvement = ((m1_mean_var - m2_mean_var) / m1_mean_var) * 100

print(f"M1 (Baseline) Average Action Variance: {m1_mean_var:.6f}")
print(f"M2 (SR2L) Average Action Variance: {m2_mean_var:.6f}")
print(f"SR2L Smoothness Improvement: {smoothness_improvement:.1f}%")
print(f"\n SR2L produces {smoothness_improvement:.1f}% smoother actions under noise!")
print(f" This demonstrates the effect of L_smooth = E[||π(s) - π(s+δ)||²] training")
print("="*80)
