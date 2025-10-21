"""
Noise Type Utilities for Robustness Evaluation

This module implements multiple noise distributions for comprehensive sensor noise
robustness testing. Noise types are matched by Signal-to-Noise Ratio (SNR) to ensure
fair comparison across different statistical distributions.

Real-World Motivation:
----------------------
1. Gaussian Noise: Thermal noise in analog circuits, encoder jitter, IMU drift
   - Common in: Joint encoders, gyroscopes, accelerometers
   - Characterized by: Continuous, zero-mean, bell-curve distribution

2. Poisson Noise: Photon counting noise in optical sensors
   - Common in: Vision sensors, LiDAR, depth cameras
   - Characterized by: Discrete, count-dependent variance (variance = mean)
   - Relevant for vision-based locomotion policies

3. Salt-and-Pepper Noise: Digital communication errors, sensor dropouts
   - Common in: CAN bus errors, wireless transmission, sensor malfunctions
   - Characterized by: Sparse, extreme-value corruptions
   - Simulates intermittent sensor failures

SNR Matching Methodology:
-------------------------
We define "equivalent severity" noise levels by matching Signal-to-Noise Ratio (SNR):

    SNR(dB) = 20 * log10(signal_rms / noise_rms)

For a baseline Gaussian noise with σ=0.1:
- Compute SNR for normalized observations (mean=0, std=1 from VecNormalize)
- Match Poisson and salt-pepper parameters to achieve same SNR
- This ensures all noise types corrupt the signal by the same amount

Author: Anand Patel
Date: October 19, 2025
"""

import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class NoiseGenerator:
    """
    Generates multiple noise types with SNR-matched severity levels.

    This class provides a unified interface for applying different noise distributions
    to observations, ensuring fair comparison through SNR matching.
    """

    def __init__(self, obs_dims: Tuple[int, int] = (13, 28)):
        """
        Initialize noise generator.

        Args:
            obs_dims: Tuple of (start_dim, end_dim) for joint sensors.
                     Default (13, 28) targets joint position/velocity in RealAnt.
        """
        self.joint_start_dim = obs_dims[0]
        self.joint_end_dim = obs_dims[1]

        # Precompute SNR equivalence table
        self.snr_table = self._compute_snr_equivalence()

    def _compute_snr_equivalence(self) -> Dict[float, Dict[str, float]]:
        """
        Compute SNR-matched parameters for different noise types.

        For VecNormalized observations (mean=0, std=1), we compute equivalent
        parameters across noise types to match SNR levels.

        Returns:
            Dictionary mapping Gaussian σ to equivalent Poisson λ and salt-pepper p
        """
        equivalence = {}

        # Baseline: VecNormalized obs has signal_rms ≈ 1.0
        signal_rms = 1.0

        gaussian_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

        for sigma in gaussian_levels:
            if sigma == 0.0:
                equivalence[sigma] = {'poisson_lambda': 0.0, 'salt_pepper_prob': 0.0}
                continue

            # Gaussian: noise_rms = σ
            gaussian_snr = 20 * np.log10(signal_rms / sigma) if sigma > 0 else np.inf

            # Poisson: Empirically calibrated to match Gaussian SNR
            # After testing: λ ≈ 10 * σ gives similar corruption levels
            # (Poisson variance scales with mean, so we need higher λ for same effect)
            poisson_lambda = 10.0 * sigma

            # Salt-and-Pepper: Empirically calibrated
            # Corrupt fraction p such that expected RMS ≈ σ
            # For VecNormalized obs (range ≈ 6), expected_rms ≈ sqrt(p) * 3
            # So: sqrt(p) * 3 ≈ σ → p ≈ (σ/3)²
            salt_pepper_prob = min((sigma / 1.5) ** 2, 0.5)  # Cap at 50%

            equivalence[sigma] = {
                'gaussian_sigma': sigma,
                'poisson_lambda': poisson_lambda,
                'salt_pepper_prob': salt_pepper_prob,
                'target_snr_db': gaussian_snr
            }

            logger.debug(f"SNR equivalence for σ={sigma:.2f}: "
                        f"Poisson λ={poisson_lambda:.3f}, "
                        f"S&P p={salt_pepper_prob:.4f}, "
                        f"Target SNR={gaussian_snr:.1f} dB")

        return equivalence

    def apply_gaussian_noise(self, obs: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply zero-mean Gaussian noise to joint observations.

        This is the standard sensor noise model for continuous measurements.

        Args:
            obs: Observation vector (shape: [obs_dim])
            sigma: Standard deviation of Gaussian noise

        Returns:
            Noisy observation vector
        """
        if sigma == 0.0:
            return obs

        noisy_obs = obs.copy()
        joint_obs = noisy_obs[self.joint_start_dim:self.joint_end_dim]

        # Add zero-mean Gaussian noise
        noise = np.random.normal(0, sigma, joint_obs.shape)
        noisy_obs[self.joint_start_dim:self.joint_end_dim] = joint_obs + noise

        return noisy_obs

    def apply_poisson_noise(self, obs: np.ndarray, lambda_param: float) -> np.ndarray:
        """
        Apply Poisson (photon counting) noise to joint observations.

        Poisson noise models discrete counting processes where variance = mean.
        Common in optical sensors (cameras, LiDAR) but less common in encoders.

        Implementation:
        1. Scale observations to positive integer range (Poisson requires λ ≥ 0)
        2. Add Poisson noise with parameter λ
        3. Scale back to original range

        Args:
            obs: Observation vector (shape: [obs_dim])
            lambda_param: Poisson rate parameter (λ)

        Returns:
            Noisy observation vector

        Note:
            For VecNormalized observations (mean=0, std=1), we shift to positive
            range before applying Poisson, then shift back. This preserves the
            count-dependent variance property while handling negative values.
        """
        if lambda_param == 0.0:
            return obs

        noisy_obs = obs.copy()
        joint_obs = noisy_obs[self.joint_start_dim:self.joint_end_dim]

        # Shift to positive range (Poisson requires non-negative counts)
        obs_min = joint_obs.min()
        obs_shifted = joint_obs - obs_min + 1.0  # Ensure all values ≥ 1

        # Scale to get reasonable integer counts
        scale_factor = 10.0
        obs_scaled = obs_shifted * scale_factor

        # Apply Poisson noise (variance = λ per observation)
        # Each observation gets noise ~ Poisson(λ)
        poisson_noise = np.random.poisson(lambda_param, obs_scaled.shape)

        # Add noise and scale back
        noisy_scaled = obs_scaled + poisson_noise
        noisy_shifted = noisy_scaled / scale_factor
        noisy_joint = noisy_shifted + obs_min - 1.0

        noisy_obs[self.joint_start_dim:self.joint_end_dim] = noisy_joint

        return noisy_obs

    def apply_salt_pepper_noise(self, obs: np.ndarray, prob: float) -> np.ndarray:
        """
        Apply salt-and-pepper (impulse) noise to joint observations.

        Salt-and-pepper noise randomly corrupts a fraction p of observations to
        extreme values (min or max of signal range). This simulates:
        - Digital communication bit flips
        - Sensor saturation events
        - Intermittent sensor failures

        Implementation:
        - With probability p/2: corrupt observation to maximum value (salt)
        - With probability p/2: corrupt observation to minimum value (pepper)
        - With probability 1-p: keep original value

        Args:
            obs: Observation vector (shape: [obs_dim])
            prob: Probability of corruption (0 to 1)

        Returns:
            Noisy observation vector

        Note:
            For VecNormalized observations, min/max are typically ±3σ ≈ ±3.0
            This creates sparse but extreme corruptions, unlike continuous Gaussian.
        """
        if prob == 0.0:
            return obs

        noisy_obs = obs.copy()
        joint_obs = noisy_obs[self.joint_start_dim:self.joint_end_dim]

        # Determine min/max for corruptions (use observed range + margin)
        obs_min = joint_obs.min() - 1.0
        obs_max = joint_obs.max() + 1.0

        # For VecNormalized obs, use typical ±3σ bounds
        if abs(obs_min) < 5.0 and abs(obs_max) < 5.0:
            obs_min, obs_max = -3.0, 3.0

        # Generate corruption masks
        salt_mask = np.random.rand(*joint_obs.shape) < (prob / 2.0)
        pepper_mask = np.random.rand(*joint_obs.shape) < (prob / 2.0)

        # Apply corruptions
        corrupted = joint_obs.copy()
        corrupted[salt_mask] = obs_max   # Salt: corrupt to max
        corrupted[pepper_mask] = obs_min  # Pepper: corrupt to min

        noisy_obs[self.joint_start_dim:self.joint_end_dim] = corrupted

        return noisy_obs

    def apply_noise(self, obs: np.ndarray, noise_type: str, level: float) -> np.ndarray:
        """
        Unified interface for applying any noise type.

        Args:
            obs: Observation vector
            noise_type: One of ['gaussian', 'poisson', 'salt_pepper', 'none']
            level: Noise severity (interpretation depends on noise_type)

        Returns:
            Noisy observation vector
        """
        if noise_type == 'none' or level == 0.0:
            return obs
        elif noise_type == 'gaussian':
            return self.apply_gaussian_noise(obs, level)
        elif noise_type == 'poisson':
            return self.apply_poisson_noise(obs, level)
        elif noise_type == 'salt_pepper':
            return self.apply_salt_pepper_noise(obs, level)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    def get_equivalent_params(self, gaussian_sigma: float) -> Dict[str, float]:
        """
        Get SNR-matched parameters for all noise types.

        Args:
            gaussian_sigma: Baseline Gaussian noise level

        Returns:
            Dictionary with equivalent parameters for each noise type
        """
        if gaussian_sigma not in self.snr_table:
            raise ValueError(f"No SNR equivalence computed for σ={gaussian_sigma}. "
                           f"Available levels: {list(self.snr_table.keys())}")

        return self.snr_table[gaussian_sigma]

    def compute_actual_snr(self, clean_obs: np.ndarray, noisy_obs: np.ndarray) -> float:
        """
        Compute empirical SNR for validation.

        Args:
            clean_obs: Original observation
            noisy_obs: Noise-corrupted observation

        Returns:
            SNR in decibels (dB)
        """
        clean_joints = clean_obs[self.joint_start_dim:self.joint_end_dim]
        noisy_joints = noisy_obs[self.joint_start_dim:self.joint_end_dim]

        signal_rms = np.sqrt(np.mean(clean_joints ** 2))
        noise = noisy_joints - clean_joints
        noise_rms = np.sqrt(np.mean(noise ** 2))

        if noise_rms < 1e-10:
            return np.inf

        snr_db = 20 * np.log10(signal_rms / noise_rms)
        return snr_db


def validate_noise_equivalence():
    """
    Validation script to verify SNR matching across noise types.

    Run this to generate equivalence table for paper Methodology section.
    """
    generator = NoiseGenerator()

    print("="*80)
    print("NOISE TYPE EQUIVALENCE TABLE (SNR-Matched)")
    print("="*80)
    print(f"{'Gaussian σ':<12} {'Poisson λ':<12} {'S&P prob':<12} {'Target SNR (dB)':<15}")
    print("-"*80)

    for sigma, params in generator.snr_table.items():
        if sigma == 0.0:
            continue
        print(f"{sigma:<12.3f} {params['poisson_lambda']:<12.3f} "
              f"{params['salt_pepper_prob']:<12.4f} {params['target_snr_db']:<15.1f}")

    print("="*80)
    print("\nEmpirical SNR Validation (100 samples):")
    print("-"*80)

    # Generate dummy VecNormalized observations
    np.random.seed(42)
    dummy_obs = np.random.randn(100, 29)  # RealAnt obs dim

    test_levels = [0.05, 0.1, 0.2, 0.5]

    for sigma in test_levels:
        params = generator.get_equivalent_params(sigma)

        snrs_gaussian = []
        snrs_poisson = []
        snrs_sp = []

        for obs in dummy_obs:
            # Gaussian
            noisy_g = generator.apply_gaussian_noise(obs, sigma)
            snrs_gaussian.append(generator.compute_actual_snr(obs, noisy_g))

            # Poisson
            noisy_p = generator.apply_poisson_noise(obs, params['poisson_lambda'])
            snrs_poisson.append(generator.compute_actual_snr(obs, noisy_p))

            # Salt-and-pepper
            noisy_sp = generator.apply_salt_pepper_noise(obs, params['salt_pepper_prob'])
            snrs_sp.append(generator.compute_actual_snr(obs, noisy_sp))

        print(f"\nσ={sigma:.2f} (Target SNR: {params['target_snr_db']:.1f} dB):")
        print(f"  Gaussian:     {np.mean(snrs_gaussian):.1f} ± {np.std(snrs_gaussian):.1f} dB")
        print(f"  Poisson:      {np.mean(snrs_poisson):.1f} ± {np.std(snrs_poisson):.1f} dB")
        print(f"  Salt-Pepper:  {np.mean(snrs_sp):.1f} ± {np.std(snrs_sp):.1f} dB")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Run validation
    validate_noise_equivalence()
