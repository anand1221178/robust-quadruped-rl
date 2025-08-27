"""PPO with regularization"""

import torch
import numpy as np
from typing import Dict, Optional, Any
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance

class PPO_SR2L(PPO):
    def __init__(
            self,
            policy,
            env: GymEnv,
            sr2l_config: Optional[Dict[str, Any]] = None,
            **kwargs
        ):
        
        super().__init__(policy, env, **kwargs)

        default_sr2l_config = {
            'enabled' :True,
            'lambda' : 0.01,
            'perturbation_std' : 0.05,
            'apply_frequency' : 1,
            'target_dimensions' : 'all',
            'warmup_steps': 0,
            'max_perturbation' : 0.2,
            'log_smoothness_metrics': True
        }

        #merge configs:
        self.sr2l_config = {**default_sr2l_config, **(sr2l_config or {})}

        #State tracking:
        self.sr2l_update_counter = 0
        self.sr2l_loss_history = []
        self.sr2l_smoothness_score = 0.0

        if self.sr2l_config['enabled']:
            print(f"SR2L enabled: lambda = {self.sr2l_config['lambda']}, perturbation_std = {self.sr2l_config['perturbation_std']}")
        else:
            print("SR2L disabled - using standard PPO!")
    
    def compute_sr2l_loss(self, observations: torch.Tensor) -> torch.Tensor:
        """
        SR2L for SENSOR NOISE robustness (per research proposal)
        L_smooth = E[||π(s) - π(s + δ)||²] where δ ~ N(0, σ²I)
        
        Handles noisy proprioceptive signals by training policy to be robust
        to sensor noise/degradation. Robot learns consistent actions despite
        noisy sensor readings.
        """
        if not self.sr2l_config['enabled']:
            return torch.tensor(0.0, device=observations.device)
        
        batch_size = observations.shape[0]

        # Get original actions from current policy
        with torch.no_grad():
            original_actions, _, _ = self.policy(observations)

        # Generate realistic sensor noise (research proposal: sensor degradation)
        noise = torch.randn_like(observations) * self.sr2l_config['perturbation_std']

        # Clamp perturbations to realistic sensor noise levels
        if self.sr2l_config.get('max_perturbation', 0) > 0:
            noise = torch.clamp(noise, -self.sr2l_config['max_perturbation'], self.sr2l_config['max_perturbation'])

        # Create perturbed observations (noisy sensor readings)
        perturbed_observations = observations + noise

        # Get actions for perturbed observations (requires gradient)
        perturbed_actions, _, _ = self.policy(perturbed_observations)

        # Compute L2 smoothness loss: ||π(s) - π(s + δ)||²
        action_diff = original_actions - perturbed_actions
        sr2l_loss = torch.mean(torch.sum(action_diff ** 2, dim=-1))

        # Track smoothness score for logging
        with torch.no_grad():
            self.sr2l_smoothness_score = 1.0 / (1.0 + sr2l_loss.item())

        return sr2l_loss
    
    def train(self) -> None:
        """Override PPO train method to apply equation 3.3: L_total = L_PPO + λ · L_smooth"""
        
        self.policy.set_training_mode(True)

        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        sr2l_losses = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                if isinstance(self.action_space, torch.Tensor):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )

                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO clipped surrogate loss
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values)

                # Entropy loss
                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                ppo_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                total_loss = ppo_loss
                sr2l_loss_value = 0.0

                # Add SR2L regularization if enabled and conditions are met
                if (self.sr2l_config['enabled'] and 
                    self.sr2l_update_counter >= self.sr2l_config['warmup_steps'] and 
                    self.sr2l_update_counter % self.sr2l_config['apply_frequency'] == 0):
                    
                    sr2l_loss = self.compute_sr2l_loss(rollout_data.observations)
                    
                    # Adaptive SR2L - reduce regularization if policy is struggling
                    adaptive_lambda = self.sr2l_config['lambda']
                    if self.sr2l_config.get('velocity_adaptive', False):
                        # Scale SR2L based on recent performance - less SR2L if struggling
                        adaptive_lambda = self.sr2l_config['lambda'] * 0.1  # Much gentler
                    
                    total_loss = ppo_loss + adaptive_lambda * sr2l_loss
                    sr2l_loss_value = sr2l_loss.item()
                    sr2l_losses.append(sr2l_loss_value)

                self.policy.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max KL")
                    break

                self.sr2l_update_counter += 1

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), 
                                         self.rollout_buffer.returns.flatten())

        # Log standard PPO metrics
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        
        # Log SR2L specific metrics
        if self.sr2l_config['enabled'] and sr2l_losses:
            self.logger.record("sr2l/loss", np.mean(sr2l_losses))
            self.logger.record("sr2l/smoothness_score", self.sr2l_smoothness_score)
            self.logger.record("sr2l/lambda", self.sr2l_config['lambda'])
            self.logger.record("sr2l/perturbation_std", self.sr2l_config['perturbation_std'])
            self.logger.record("sr2l/update_counter", self.sr2l_update_counter)
        
        # Log policy standard deviation if available
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())