# evaluation.py - Reusable evaluation script
"""
Evaluation script for robust quadruped locomotion
Use this to evaluate any trained policy
"""

from define_success_metrics import MetricsEvaluator, SuccessMetrics
import numpy as np

def evaluate_policy(policy_function, n_episodes=100, verbose=True):
    """
    Evaluate a trained policy
    
    Args:
        policy_function: Function that takes obs -> action
        n_episodes: Number of evaluation episodes
        verbose: Print results
        
    Returns:
        Dictionary of results
    """
    evaluator = MetricsEvaluator()
    results = evaluator.evaluate_batch(n_episodes, policy_function)
    
    if verbose:
        print(f"\nEvaluation over {n_episodes} episodes:")
        print(f"  Success rate: {results['success_rate']*100:.1f}%")
        print(f"  Avg reward: {results['avg_cumulative_reward']:.2f}")
        print(f"  Avg distance: {results['avg_distance']:.2f} m")
        print(f"  Avg velocity: {results['avg_velocity']:.2f} m/s")
    
    return results

# Example usage:
# from stable_baselines3 import PPO
# model = PPO.load("models/ppo_baseline.zip")
# results = evaluate_policy(lambda obs: model.predict(obs)[0])
