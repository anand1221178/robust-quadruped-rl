#!/usr/bin/env python3
"""
Gradient Conflict Visualization for SR2L vs PPO/DR

Generates publication-quality figures showing:
1. 3D loss landscape surfaces (PPO vs SR2L) in opposing directions
2. Gradient cosine similarity showing angular conflict
3. 2D vector field showing push-pull between objectives

Inspired by Ilyas et al. (2018) "Are Deep Policy Gradient Algorithms
Truly Policy Gradient Algorithms?" and Li et al. (2018) loss landscape
visualization using filter-normalized random directions.

Usage:
    python scripts/visualize_gradient_conflict.py \
        --model experiments/M4_combo_v2_seed42/final_model.zip \
        --vec experiments/M4_combo_v2_seed42/vec_normalize.pkl \
        --output figures/gradient_conflict

    # Or use any M2/M4 model (needs SR2L enabled):
    python scripts/visualize_gradient_conflict.py \
        --model done/ultimate_robustness_combo_ju7lfsk2/final_model.zip \
        --vec done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl \
        --output figures/gradient_conflict
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from copy import deepcopy

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from agents.ppo_sr2l import PPO_SR2L
import realant_sim


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize gradient conflict between PPO and SR2L')
    parser.add_argument('--model', type=str, required=True, help='Path to model .zip file')
    parser.add_argument('--vec', type=str, required=True, help='Path to vec_normalize.pkl')
    parser.add_argument('--output', type=str, default='figures/gradient_conflict',
                        help='Output directory for figures')
    parser.add_argument('--n-samples', type=int, default=2048,
                        help='Number of observation samples for gradient computation')
    parser.add_argument('--grid-size', type=int, default=31,
                        help='Grid resolution for loss landscape (31x31 default)')
    parser.add_argument('--landscape-range', type=float, default=2.0,
                        help='Range of perturbation in parameter space')
    parser.add_argument('--sr2l-lambda', type=float, default=0.001,
                        help='SR2L regularization weight')
    parser.add_argument('--sr2l-sigma', type=float, default=0.01,
                        help='SR2L perturbation std')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_model_and_env(model_path, vec_path):
    """Load trained model and normalized environment."""
    env = DummyVecEnv([lambda: Monitor(gym.make("RealAntMujoco-v0"))])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    env.norm_reward = False

    # Try loading as PPO_SR2L first, fall back to PPO
    try:
        model = PPO_SR2L.load(model_path, env=env)
        print(f"Loaded as PPO_SR2L")
    except Exception:
        model = PPO.load(model_path, env=env)
        print(f"Loaded as standard PPO")

    return model, env


def collect_observations(env, model, n_samples):
    """Collect observations from rollouts for gradient computation."""
    obs_list = []
    actions_list = []
    rewards_list = []
    old_log_probs_list = []
    values_list = []

    obs = env.reset()
    collected = 0

    while collected < n_samples:
        action, _ = model.predict(obs, deterministic=False)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(model.device)
            dist = model.policy.get_distribution(obs_tensor)
            action_tensor = torch.FloatTensor(action).to(model.device)
            log_prob = dist.log_prob(action_tensor)
            value = model.policy.predict_values(obs_tensor)

        obs_list.append(obs.copy())
        actions_list.append(action.copy())
        old_log_probs_list.append(log_prob.cpu().numpy())
        values_list.append(value.cpu().numpy())

        obs, reward, done, info = env.step(action)
        rewards_list.append(reward)
        collected += 1

    return {
        'observations': np.concatenate(obs_list, axis=0),
        'actions': np.concatenate(actions_list, axis=0),
        'rewards': np.concatenate(rewards_list, axis=0),
        'old_log_probs': np.concatenate(old_log_probs_list, axis=0),
        'values': np.concatenate(values_list, axis=0),
    }


def compute_ppo_loss(model, obs_tensor, actions_tensor, old_log_probs_tensor,
                     advantages_tensor, clip_range=0.2):
    """Compute PPO clipped surrogate loss."""
    values, log_prob, entropy = model.policy.evaluate_actions(obs_tensor, actions_tensor)
    ratio = torch.exp(log_prob - old_log_probs_tensor)
    policy_loss_1 = advantages_tensor * ratio
    policy_loss_2 = advantages_tensor * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    return policy_loss


def compute_sr2l_loss(model, obs_tensor, sigma=0.01):
    """Compute SR2L smoothness loss: E[||pi(s) - pi(s+delta)||^2]."""
    original_actions, _, _ = model.policy(obs_tensor)

    perturbed_obs = obs_tensor.clone()
    noise = torch.randn(obs_tensor.shape[0], 16, device=obs_tensor.device) * sigma
    noise = torch.clamp(noise, -0.05, 0.05)
    perturbed_obs[:, 13:29] = obs_tensor[:, 13:29] + noise

    perturbed_actions, _, _ = model.policy(perturbed_obs)

    action_diff = original_actions - perturbed_actions
    sr2l_loss = torch.mean(torch.sum(action_diff ** 2, dim=-1))
    return sr2l_loss


def get_random_directions(model):
    """Generate two random filter-normalized directions in parameter space."""
    # Collect all policy parameters into a single vector
    params = []
    for p in model.policy.parameters():
        params.append(p.data.flatten())
    theta = torch.cat(params)

    # Generate two random directions
    d1 = torch.randn_like(theta)
    d2 = torch.randn_like(theta)

    # Filter normalization: scale each layer's direction by the layer's norm
    # This gives more meaningful visualization (Li et al. 2018)
    idx = 0
    for p in model.policy.parameters():
        n = p.numel()
        layer_norm = p.data.norm()

        d1_slice = d1[idx:idx+n].view(p.shape)
        d1_slice_norm = d1_slice.norm()
        if d1_slice_norm > 0:
            d1[idx:idx+n] = (d1_slice / d1_slice_norm * layer_norm).flatten()

        d2_slice = d2[idx:idx+n].view(p.shape)
        d2_slice_norm = d2_slice.norm()
        if d2_slice_norm > 0:
            d2[idx:idx+n] = (d2_slice / d2_slice_norm * layer_norm).flatten()

        idx += n

    return d1, d2


def set_params_from_vector(model, param_vector):
    """Set model parameters from a flattened vector."""
    idx = 0
    for p in model.policy.parameters():
        n = p.numel()
        p.data = param_vector[idx:idx+n].view(p.shape)
        idx += n


def get_param_vector(model):
    """Get model parameters as a flattened vector."""
    params = []
    for p in model.policy.parameters():
        params.append(p.data.flatten())
    return torch.cat(params)


def compute_gradient_cosine_similarity(model, data, sr2l_lambda=0.001, sr2l_sigma=0.01):
    """
    Compute cosine similarity between PPO and SR2L gradients.
    Negative values = opposing directions (conflict).
    """
    obs_tensor = torch.FloatTensor(data['observations']).to(model.device)
    actions_tensor = torch.FloatTensor(data['actions']).to(model.device)
    old_log_probs_tensor = torch.FloatTensor(data['old_log_probs']).to(model.device)

    # Simple advantages (rewards - mean) / std
    rewards = data['rewards']
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    advantages_tensor = torch.FloatTensor(advantages).to(model.device)

    # Compute PPO gradient
    model.policy.optimizer.zero_grad()
    ppo_loss = compute_ppo_loss(model, obs_tensor, actions_tensor,
                                old_log_probs_tensor, advantages_tensor)
    ppo_loss.backward(retain_graph=True)
    ppo_grad = torch.cat([p.grad.flatten() for p in model.policy.parameters()
                          if p.grad is not None])

    # Compute SR2L gradient
    model.policy.optimizer.zero_grad()
    sr2l_loss = compute_sr2l_loss(model, obs_tensor, sigma=sr2l_sigma)
    (sr2l_lambda * sr2l_loss).backward()
    sr2l_grad = torch.cat([p.grad.flatten() for p in model.policy.parameters()
                           if p.grad is not None])

    model.policy.optimizer.zero_grad()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        ppo_grad.unsqueeze(0), sr2l_grad.unsqueeze(0)
    ).item()

    return cos_sim, ppo_grad.cpu().numpy(), sr2l_grad.cpu().numpy(), \
           ppo_loss.item(), sr2l_loss.item()


def compute_loss_landscapes(model, data, d1, d2, grid_size=31,
                            landscape_range=2.0, sr2l_lambda=0.001,
                            sr2l_sigma=0.01):
    """Compute PPO, SR2L, and combined loss on a 2D grid in parameter space."""
    theta_star = get_param_vector(model).clone()

    alphas = np.linspace(-landscape_range, landscape_range, grid_size)
    betas = np.linspace(-landscape_range, landscape_range, grid_size)

    ppo_landscape = np.zeros((grid_size, grid_size))
    sr2l_landscape = np.zeros((grid_size, grid_size))
    combined_landscape = np.zeros((grid_size, grid_size))

    obs_tensor = torch.FloatTensor(data['observations']).to(model.device)
    actions_tensor = torch.FloatTensor(data['actions']).to(model.device)
    old_log_probs_tensor = torch.FloatTensor(data['old_log_probs']).to(model.device)

    rewards = data['rewards']
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    advantages_tensor = torch.FloatTensor(advantages).to(model.device)

    total_points = grid_size * grid_size
    print(f"Computing loss landscape ({grid_size}x{grid_size} = {total_points} points)...")

    for i, alpha in enumerate(alphas):
        if i % 5 == 0:
            print(f"  Row {i+1}/{grid_size}...")
        for j, beta in enumerate(betas):
            # Perturb parameters: theta = theta* + alpha*d1 + beta*d2
            theta_perturbed = theta_star + alpha * d1 + beta * d2
            set_params_from_vector(model, theta_perturbed)

            with torch.no_grad():
                # PPO loss (we need gradients=False for landscape)
                try:
                    values, log_prob, entropy = model.policy.evaluate_actions(
                        obs_tensor, actions_tensor)
                    ratio = torch.exp(log_prob - old_log_probs_tensor)
                    pl1 = advantages_tensor * ratio
                    pl2 = advantages_tensor * torch.clamp(ratio, 0.8, 1.2)
                    ppo_loss = -torch.min(pl1, pl2).mean().item()
                except Exception:
                    ppo_loss = float('nan')

                # SR2L loss
                try:
                    original_actions, _, _ = model.policy(obs_tensor)
                    perturbed_obs = obs_tensor.clone()
                    noise = torch.randn(obs_tensor.shape[0], 16,
                                        device=obs_tensor.device) * sr2l_sigma
                    noise = torch.clamp(noise, -0.05, 0.05)
                    perturbed_obs[:, 13:29] = obs_tensor[:, 13:29] + noise
                    perturbed_actions, _, _ = model.policy(perturbed_obs)
                    diff = original_actions - perturbed_actions
                    sr2l_loss = torch.mean(torch.sum(diff ** 2, dim=-1)).item()
                except Exception:
                    sr2l_loss = float('nan')

            ppo_landscape[i, j] = ppo_loss
            sr2l_landscape[i, j] = sr2l_loss
            combined_landscape[i, j] = ppo_loss + sr2l_lambda * sr2l_loss

    # Restore original parameters
    set_params_from_vector(model, theta_star)

    return alphas, betas, ppo_landscape, sr2l_landscape, combined_landscape


def compute_gradient_field(model, data, d1, d2, field_size=9,
                           field_range=1.5, sr2l_lambda=0.001,
                           sr2l_sigma=0.01):
    """Compute PPO and SR2L gradient vectors on a 2D grid."""
    theta_star = get_param_vector(model).clone()

    alphas = np.linspace(-field_range, field_range, field_size)
    betas = np.linspace(-field_range, field_range, field_size)

    ppo_grad_field = np.zeros((field_size, field_size, 2))
    sr2l_grad_field = np.zeros((field_size, field_size, 2))

    obs_tensor = torch.FloatTensor(data['observations']).to(model.device)
    actions_tensor = torch.FloatTensor(data['actions']).to(model.device)
    old_log_probs_tensor = torch.FloatTensor(data['old_log_probs']).to(model.device)

    rewards = data['rewards']
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    advantages_tensor = torch.FloatTensor(advantages).to(model.device)

    print(f"Computing gradient field ({field_size}x{field_size})...")

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            theta_perturbed = theta_star + alpha * d1 + beta * d2
            set_params_from_vector(model, theta_perturbed)

            # PPO gradient projected onto d1, d2
            model.policy.optimizer.zero_grad()
            try:
                ppo_loss = compute_ppo_loss(model, obs_tensor, actions_tensor,
                                            old_log_probs_tensor, advantages_tensor)
                ppo_loss.backward()
                grad = torch.cat([p.grad.flatten() for p in model.policy.parameters()
                                  if p.grad is not None])
                ppo_grad_field[i, j, 0] = -torch.dot(grad, d1).item()  # negative = descent
                ppo_grad_field[i, j, 1] = -torch.dot(grad, d2).item()
            except Exception:
                pass

            # SR2L gradient projected onto d1, d2
            model.policy.optimizer.zero_grad()
            try:
                sr2l_loss = compute_sr2l_loss(model, obs_tensor, sigma=sr2l_sigma)
                (sr2l_lambda * sr2l_loss).backward()
                grad = torch.cat([p.grad.flatten() for p in model.policy.parameters()
                                  if p.grad is not None])
                sr2l_grad_field[i, j, 0] = -torch.dot(grad, d1).item()
                sr2l_grad_field[i, j, 1] = -torch.dot(grad, d2).item()
            except Exception:
                pass

    model.policy.optimizer.zero_grad()
    set_params_from_vector(model, theta_star)

    return alphas, betas, ppo_grad_field, sr2l_grad_field


def plot_3d_landscapes(alphas, betas, ppo_landscape, sr2l_landscape,
                       combined_landscape, output_dir):
    """Plot 3D loss landscape surfaces - the main publication figure."""
    A, B = np.meshgrid(alphas, betas, indexing='ij')

    fig = plt.figure(figsize=(18, 5.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.25)

    # Normalize for comparable visualization
    def safe_normalize(arr):
        arr = np.nan_to_num(arr, nan=np.nanmean(arr))
        vmin, vmax = np.nanpercentile(arr, [2, 98])
        return arr, vmin, vmax

    titles = [
        r'$\nabla_\theta \mathcal{L}_{PPO}$: Policy Gradient Loss',
        r'$\nabla_\theta \mathcal{L}_{SR2L}$: Smoothness Loss',
        r'$\mathcal{L}_{total} = \mathcal{L}_{PPO} + \lambda \cdot \mathcal{L}_{SR2L}$'
    ]
    landscapes = [ppo_landscape, sr2l_landscape, combined_landscape]
    cmaps = [cm.RdYlBu_r, cm.YlGn_r, cm.viridis]

    for idx, (landscape, title, cmap) in enumerate(zip(landscapes, titles, cmaps)):
        ax = fig.add_subplot(gs[idx], projection='3d')
        data, vmin, vmax = safe_normalize(landscape)

        surf = ax.plot_surface(A, B, data, cmap=cmap, alpha=0.85,
                               edgecolor='grey', linewidth=0.15,
                               antialiased=True, rcount=50, ccount=50)

        ax.set_xlabel('Step direction', fontsize=9, labelpad=5)
        ax.set_ylabel('Random direction', fontsize=9, labelpad=5)
        ax.set_zlabel('Loss', fontsize=9, labelpad=3)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=25, azim=-60)

        # Mark optimum (center)
        ax.scatter([0], [0], [data[len(alphas)//2, len(betas)//2]],
                   color='red', s=50, zorder=10, marker='*')

    plt.suptitle('Loss Landscape Visualization: Opposing Optimization Surfaces',
                 fontsize=13, fontweight='bold', y=1.02)

    path = os.path.join(output_dir, 'loss_landscape_3d.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_gradient_vector_field(alphas, betas, ppo_grad_field, sr2l_grad_field,
                               output_dir):
    """Plot 2D gradient vector field showing opposing directions."""
    A, B = np.meshgrid(alphas, betas, indexing='ij')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Normalize arrow lengths for visibility
    def normalize_field(field):
        magnitudes = np.sqrt(field[:,:,0]**2 + field[:,:,1]**2)
        max_mag = np.percentile(magnitudes[magnitudes > 0], 95) if np.any(magnitudes > 0) else 1
        if max_mag > 0:
            field = field / max_mag
        return field

    ppo_norm = normalize_field(ppo_grad_field.copy())
    sr2l_norm = normalize_field(sr2l_grad_field.copy())

    # Panel 1: PPO gradients
    ax = axes[0]
    ax.quiver(A, B, ppo_norm[:,:,0], ppo_norm[:,:,1],
              color='#2196F3', alpha=0.8, scale=15, width=0.004)
    ax.set_title(r'$-\nabla_\theta \mathcal{L}_{PPO}$' + '\n(Policy improvement direction)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Step direction')
    ax.set_ylabel('Random direction')
    ax.scatter([0], [0], color='red', s=80, zorder=10, marker='*')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Panel 2: SR2L gradients
    ax = axes[1]
    ax.quiver(A, B, sr2l_norm[:,:,0], sr2l_norm[:,:,1],
              color='#FF5722', alpha=0.8, scale=15, width=0.004)
    ax.set_title(r'$-\nabla_\theta \mathcal{L}_{SR2L}$' + '\n(Smoothness direction)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Step direction')
    ax.set_ylabel('Random direction')
    ax.scatter([0], [0], color='red', s=80, zorder=10, marker='*')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Panel 3: Both overlaid
    ax = axes[2]
    ax.quiver(A, B, ppo_norm[:,:,0], ppo_norm[:,:,1],
              color='#2196F3', alpha=0.6, scale=15, width=0.004,
              label=r'$-\nabla \mathcal{L}_{PPO}$')
    ax.quiver(A, B, sr2l_norm[:,:,0], sr2l_norm[:,:,1],
              color='#FF5722', alpha=0.6, scale=15, width=0.004,
              label=r'$-\nabla \mathcal{L}_{SR2L}$')
    ax.set_title('Gradient Conflict\n(Opposing directions = interference)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Step direction')
    ax.set_ylabel('Random direction')
    ax.scatter([0], [0], color='red', s=80, zorder=10, marker='*')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')

    # Compute and annotate cosine similarity at each point
    cos_sims = []
    for i in range(len(alphas)):
        for j in range(len(betas)):
            pv = ppo_grad_field[i, j]
            sv = sr2l_grad_field[i, j]
            pn = np.linalg.norm(pv)
            sn = np.linalg.norm(sv)
            if pn > 0 and sn > 0:
                cos_sims.append(np.dot(pv, sv) / (pn * sn))
    if cos_sims:
        mean_cos = np.mean(cos_sims)
        ax.text(0.02, 0.02,
                f'Mean cos similarity: {mean_cos:.3f}\n(negative = conflict)',
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('Gradient Direction Analysis: PPO vs SR2L',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'gradient_vector_field.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_cosine_similarity_summary(cos_sim, ppo_loss, sr2l_loss,
                                   ppo_grad, sr2l_grad, output_dir):
    """Plot summary figure with gradient statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Cosine similarity gauge
    ax = axes[0]
    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax.fill_between(np.cos(theta[:50]), 0, np.sin(theta[:50]),
                    alpha=0.15, color='green', label='Cooperative')
    ax.fill_between(np.cos(theta[50:]), 0, np.sin(theta[50:]),
                    alpha=0.15, color='red', label='Conflicting')

    # Draw the actual angle
    angle = np.arccos(np.clip(cos_sim, -1, 1))
    ax.arrow(0, 0, 0.85*np.cos(angle), 0.85*np.sin(angle),
             head_width=0.05, head_length=0.05, fc='black', ec='black', linewidth=2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.set_aspect('equal')
    ax.set_title(f'Gradient Angle: {np.degrees(angle):.1f}°\n'
                 f'cos(θ) = {cos_sim:.4f}', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='grey', linewidth=0.5)
    ax.axvline(x=0, color='grey', linewidth=0.5)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlabel('Cosine similarity')

    # Panel 2: Gradient magnitude comparison
    ax = axes[1]
    ppo_mag = np.linalg.norm(ppo_grad)
    sr2l_mag = np.linalg.norm(sr2l_grad)
    bars = ax.bar(['PPO\n(policy gradient)', 'SR2L\n(smoothness)'],
                  [ppo_mag, sr2l_mag],
                  color=['#2196F3', '#FF5722'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Gradient magnitude (L2 norm)')
    ax.set_title('Gradient Magnitudes', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, [ppo_mag, sr2l_mag]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + ppo_mag*0.02,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)

    # Panel 3: Loss values
    ax = axes[2]
    bars = ax.bar(['PPO Loss', 'SR2L Loss\n(×λ=0.001)'],
                  [abs(ppo_loss), sr2l_loss * 0.001],
                  color=['#2196F3', '#FF5722'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Loss value')
    ax.set_title('Loss Contributions', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, [abs(ppo_loss), sr2l_loss * 0.001]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + abs(ppo_loss)*0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Gradient Conflict Analysis: cos(∇L_PPO, ∇L_SR2L) = {cos_sim:.4f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'gradient_conflict_summary.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("GRADIENT CONFLICT VISUALIZATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"SR2L: lambda={args.sr2l_lambda}, sigma={args.sr2l_sigma}")

    # Load model
    print("\n1. Loading model...")
    model, env = load_model_and_env(args.model, args.vec)

    # Collect observations
    print(f"\n2. Collecting {args.n_samples} observation samples...")
    data = collect_observations(env, model, args.n_samples)
    print(f"   Observations shape: {data['observations'].shape}")
    print(f"   Mean reward: {data['rewards'].mean():.2f}")

    # Compute gradient cosine similarity at current parameters
    print("\n3. Computing gradient cosine similarity...")
    cos_sim, ppo_grad, sr2l_grad, ppo_loss, sr2l_loss = \
        compute_gradient_cosine_similarity(
            model, data, args.sr2l_lambda, args.sr2l_sigma)
    print(f"   Cosine similarity: {cos_sim:.4f}")
    print(f"   Angle: {np.degrees(np.arccos(np.clip(cos_sim, -1, 1))):.1f} degrees")
    print(f"   PPO loss: {ppo_loss:.4f}")
    print(f"   SR2L loss: {sr2l_loss:.4f}")

    if cos_sim < 0:
        print("   >>> CONFLICT DETECTED: Gradients point in opposing directions!")
    else:
        print("   >>> Gradients are cooperative (same general direction)")

    # Plot cosine similarity summary
    print("\n4. Plotting gradient conflict summary...")
    plot_cosine_similarity_summary(cos_sim, ppo_loss, sr2l_loss,
                                   ppo_grad, sr2l_grad, args.output)

    # Generate random directions for landscape visualization
    print("\n5. Generating random directions for landscape...")
    d1, d2 = get_random_directions(model)

    # Compute loss landscapes
    print("\n6. Computing loss landscapes...")
    alphas, betas, ppo_landscape, sr2l_landscape, combined_landscape = \
        compute_loss_landscapes(
            model, data, d1, d2,
            grid_size=args.grid_size,
            landscape_range=args.landscape_range,
            sr2l_lambda=args.sr2l_lambda,
            sr2l_sigma=args.sr2l_sigma)

    # Plot 3D landscapes
    print("\n7. Plotting 3D loss landscapes...")
    plot_3d_landscapes(alphas, betas, ppo_landscape, sr2l_landscape,
                       combined_landscape, args.output)

    # Compute and plot gradient vector field
    print("\n8. Computing gradient vector field...")
    field_alphas, field_betas, ppo_field, sr2l_field = \
        compute_gradient_field(
            model, data, d1, d2,
            field_size=9, field_range=1.5,
            sr2l_lambda=args.sr2l_lambda,
            sr2l_sigma=args.sr2l_sigma)

    print("\n9. Plotting gradient vector field...")
    plot_gradient_vector_field(field_alphas, field_betas,
                               ppo_field, sr2l_field, args.output)

    print("\n" + "=" * 60)
    print("DONE! Figures saved to:", args.output)
    print("=" * 60)
    print("\nFiles generated:")
    for f in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
