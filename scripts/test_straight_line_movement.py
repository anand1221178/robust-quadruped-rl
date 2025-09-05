#!/usr/bin/env python3
"""
Test if current models are walking straight or circling
Compare straight-line distance vs total distance traveled
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import matplotlib.pyplot as plt

def test_straight_line_vs_circular(model_path, episodes=5, steps=500):
    """Test if model walks straight or circles"""
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize
    norm_path = model_path.replace('/best_model.zip', '/../vec_normalize.pkl')
    try:
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"✅ Loaded normalization: {norm_path}")
    except:
        print("⚠️ No normalization file found")
    
    results = []
    
    for episode in range(episodes):
        obs = env.reset()
        
        # Track positions
        positions = []
        
        for step in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Get position
            try:
                base_env = env.venv.envs[0]
                if hasattr(base_env, 'sim') and base_env.sim is not None:
                    pos = base_env.sim.data.qpos[:3].copy()  # x, y, z
                    positions.append(pos)
            except:
                pass
            
            if done[0]:
                break
        
        if len(positions) > 1:
            positions = np.array(positions)
            
            # Calculate metrics
            start_pos = positions[0]
            end_pos = positions[-1]
            
            # Straight-line distance (start to end)
            straight_distance = np.linalg.norm(end_pos[:2] - start_pos[:2])
            
            # Total path distance (sum of all segments)
            path_distances = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
            total_distance = np.sum(path_distances)
            
            # Straight-line velocity
            straight_velocity = straight_distance / (len(positions) * 0.05)  # dt = 0.05
            
            # Path velocity (what we currently measure)
            path_velocity = total_distance / (len(positions) * 0.05)
            
            # Lateral deviation (how far from straight line)
            lateral_positions = positions[:, 1] - start_pos[1]  # y-axis deviation
            max_lateral_deviation = np.max(np.abs(lateral_positions))
            mean_lateral_deviation = np.mean(np.abs(lateral_positions))
            
            # Efficiency ratio
            efficiency = straight_distance / total_distance if total_distance > 0 else 0
            
            results.append({
                'episode': episode + 1,
                'straight_distance': straight_distance,
                'total_distance': total_distance,
                'straight_velocity': straight_velocity,
                'path_velocity': path_velocity,
                'max_lateral_deviation': max_lateral_deviation,
                'mean_lateral_deviation': mean_lateral_deviation,
                'efficiency': efficiency,
                'positions': positions
            })
            
            print(f"Episode {episode + 1}:")
            print(f"  Straight distance: {straight_distance:.2f}m")
            print(f"  Total distance: {total_distance:.2f}m") 
            print(f"  Straight velocity: {straight_velocity:.3f} m/s")
            print(f"  Path velocity: {path_velocity:.3f} m/s")
            print(f"  Max lateral deviation: {max_lateral_deviation:.2f}m")
            print(f"  Efficiency (straight/total): {efficiency:.1%}")
            print()
    
    # Summary statistics
    if results:
        straight_velocities = [r['straight_velocity'] for r in results]
        path_velocities = [r['path_velocity'] for r in results]
        efficiencies = [r['efficiency'] for r in results]
        lateral_devs = [r['mean_lateral_deviation'] for r in results]
        
        print("=" * 60)
        print(f"SUMMARY FOR {model_path}")
        print("=" * 60)
        print(f"Average straight-line velocity: {np.mean(straight_velocities):.3f} ± {np.std(straight_velocities):.3f} m/s")
        print(f"Average path velocity: {np.mean(path_velocities):.3f} ± {np.std(path_velocities):.3f} m/s")
        print(f"Average efficiency: {np.mean(efficiencies):.1%} ± {np.std(efficiencies):.1%}")
        print(f"Average lateral deviation: {np.mean(lateral_devs):.2f} ± {np.std(lateral_devs):.2f}m")
        
        # Interpretation
        avg_efficiency = np.mean(efficiencies)
        avg_lateral = np.mean(lateral_devs)
        
        print("\n" + "=" * 60)
        print("INTERPRETATION:")
        print("=" * 60)
        
        if avg_efficiency > 0.8:
            print("✅ Robot walks MOSTLY STRAIGHT")
            print("   - High efficiency suggests minimal circling")
        elif avg_efficiency > 0.6:
            print("⚠️ Robot has MODERATE path deviation")
            print("   - Some unnecessary movement but not severe circling")
        else:
            print("❌ Robot is CIRCLING or wandering significantly")
            print("   - Low efficiency suggests major path inefficiency")
        
        if avg_lateral < 0.5:
            print("✅ Low lateral deviation - stays close to straight line")
        elif avg_lateral < 1.0:
            print("⚠️ Moderate lateral deviation - some side-to-side movement")
        else:
            print("❌ High lateral deviation - significant wandering")
        
        velocity_ratio = np.mean(straight_velocities) / np.mean(path_velocities)
        print(f"\nVelocity Impact: Straight-line speed is {velocity_ratio:.1%} of path speed")
        
        if velocity_ratio < 0.7:
            print("❌ MAJOR IMPACT: Current measurements severely overestimate straight progress")
            print("   → Straight-line constraint highly recommended")
        elif velocity_ratio < 0.85:
            print("⚠️ MODERATE IMPACT: Some overestimation of straight progress")
            print("   → Consider straight-line constraint")
        else:
            print("✅ MINIMAL IMPACT: Current measurements are reasonably accurate")
            print("   → Straight-line constraint may not be necessary")
    
    env.close()
    return results

def compare_models():
    """Compare straight-line performance across models"""
    
    models = {
        'Baseline': 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip',
        'DR v2': 'done/ppo_dr_gentle_v2_wptws01u/best_model/best_model.zip'
    }
    
    print("=" * 80)
    print("🎯 TESTING STRAIGHT-LINE vs CIRCULAR MOVEMENT")
    print("=" * 80)
    print("This test determines if robots are walking straight or circling around")
    print("High efficiency = straight walking, Low efficiency = circling/wandering")
    print()
    
    all_results = {}
    
    for name, path in models.items():
        print(f"\n{'='*20} TESTING {name.upper()} {'='*20}")
        try:
            results = test_straight_line_vs_circular(path, episodes=3, steps=300)
            all_results[name] = results
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
    
    # Final comparison
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("🏆 FINAL COMPARISON")
        print("=" * 80)
        
        for name, results in all_results.items():
            if results:
                efficiencies = [r['efficiency'] for r in results]
                straight_vels = [r['straight_velocity'] for r in results]
                print(f"{name:>12}: {np.mean(efficiencies):.1%} efficiency, "
                      f"{np.mean(straight_vels):.3f} m/s straight-line speed")
        
        print("\n💡 RECOMMENDATION:")
        
        # Check if any model has low efficiency
        low_efficiency = any(
            np.mean([r['efficiency'] for r in results]) < 0.7
            for results in all_results.values() if results
        )
        
        if low_efficiency:
            print("❌ At least one model shows significant circling/wandering")
            print("   → Straight-line constraint HIGHLY RECOMMENDED")
            print("   → Retrain models with straight-line wrapper")
        else:
            print("✅ Models generally walk straight")
            print("   → Current approach is acceptable")
            print("   → Focus on permanent DR without straight-line constraint")

if __name__ == "__main__":
    compare_models()