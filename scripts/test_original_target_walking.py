#!/usr/bin/env python3
"""
Test the ORIGINAL TargetWalkingWrapper with working baseline
Skip all the "smooth" bullshit - just test goal-directed behavior
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.target_walking_wrapper import TargetWalkingWrapper
import realant_sim

def test_original_target_walking():
    """Test TargetWalkingWrapper with working baseline"""
    
    print("🎯 TESTING ORIGINAL TargetWalkingWrapper")
    print("Using the proven working baseline + simple goal-directed wrapper")
    print("=" * 70)
    
    # Use the PROVEN working baseline
    model_path = 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip'
    norm_path = 'done/ppo_baseline_ueqbjf2x/vec_normalize.pkl'
    
    print(f"✅ Using PROVEN baseline: {model_path}")
    print(f"✅ Baseline performance: 0.214 m/s smooth walking")
    
    # Create environment with ORIGINAL TargetWalkingWrapper
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = TargetWalkingWrapper(env, target_distance=5.0)
        return env
    
    # Note: This might have observation space issues since baseline was trained without wrapper
    # But let's see what happens
    
    env = DummyVecEnv([make_env])
    
    print("\n📏 Environment Info:")
    print(f"  TargetWalkingWrapper obs space: {env.observation_space}")
    
    try:
        # Try loading VecNormalize (might fail due to obs space mismatch)
        env = VecNormalize.load(norm_path, env)
        env.training = False
        print("✅ VecNormalize loaded (obs space matched!)")
    except Exception as e:
        print(f"⚠️  VecNormalize failed (expected): {e}")
        print("  Will test without normalization")
        # Create new env without VecNormalize
        env.close()
        env = DummyVecEnv([make_env])
    
    # Load model
    try:
        model = PPO.load(model_path)
        print("✅ Baseline model loaded")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Test goal-directed behavior
    print(f"\n🤖 TESTING GOAL-DIRECTED BEHAVIOR:")
    print("-" * 50)
    
    episode_results = []
    
    for episode in range(3):
        print(f"\n📍 EPISODE {episode + 1}:")
        
        try:
            obs = env.reset()
            episode_reward = 0
            targets_reached = 0
            positions = []
            
            for step in range(500):  # Shorter test
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward[0]
                
                # Track position
                if hasattr(env.envs[0], 'unwrapped'):
                    current_x = env.envs[0].unwrapped.data.qpos[0]
                    positions.append(current_x)
                
                # Check for targets (TargetWalkingWrapper info)
                if info[0]:
                    if 'success_bonus' in info[0] and info[0]['success_bonus'] > 0:
                        targets_reached += 1
                        current_pos = positions[-1] if positions else 0
                        print(f"  🎯 TARGET {targets_reached} REACHED at x={current_pos:.1f}m (step {step})")
                
                # Progress updates
                if step % 100 == 0:
                    current_pos = positions[-1] if positions else 0
                    dist_to_target = info[0].get('distance_to_target', 'N/A') if info[0] else 'N/A'
                    print(f"  Step {step:3d}: reward={episode_reward:.0f}, x={current_pos:5.1f}m, target_dist={dist_to_target}")
                
                if done[0]:
                    print(f"  Episode ended at step {step}")
                    break
            
            # Episode results
            total_distance = positions[-1] - positions[0] if len(positions) >= 2 else 0
            avg_velocity = total_distance / (step * 0.05) if step > 0 else 0
            
            episode_results.append({
                'reward': episode_reward,
                'targets': targets_reached,
                'distance': total_distance,
                'velocity': avg_velocity,
                'steps': step + 1
            })
            
            print(f"  📊 Episode {episode + 1}: {episode_reward:.0f} reward, {targets_reached} targets, {total_distance:.1f}m, {avg_velocity:.3f} m/s")
            
        except Exception as e:
            print(f"  ❌ Episode {episode + 1} failed: {e}")
            continue
    
    # Overall assessment
    if episode_results:
        print(f"\n📈 ORIGINAL TARGET WALKING RESULTS:")
        print("=" * 70)
        
        avg_reward = np.mean([r['reward'] for r in episode_results])
        avg_targets = np.mean([r['targets'] for r in episode_results])
        avg_distance = np.mean([r['distance'] for r in episode_results])
        avg_velocity = np.mean([r['velocity'] for r in episode_results])
        
        print(f"✅ Average Episode Reward: {avg_reward:.0f}")
        print(f"✅ Average Targets Reached: {avg_targets:.1f} per episode")
        print(f"✅ Average Distance Traveled: {avg_distance:.1f} meters")
        print(f"✅ Average Velocity: {avg_velocity:.3f} m/s")
        
        # Success assessment
        print(f"\n🎯 ASSESSMENT:")
        if avg_targets >= 1.0:
            print("🎉 SUCCESS: Robot reaches targets with original TargetWalkingWrapper!")
            print("✅ Goal-directed behavior working!")
            print("🚀 Ready to fine-tune this approach for robustness!")
        elif avg_velocity > 0.15:
            print("⚠️  PARTIAL: Robot moves well but target detection needs work")
            print("📝 Could fine-tune target detection logic")
        else:
            print("❌ FAILED: Still not working - need different approach")
        
        # Compare to baseline
        baseline_vel = 0.214
        print(f"\n📊 COMPARISON TO BASELINE:")
        print(f"  Baseline velocity: {baseline_vel:.3f} m/s")
        print(f"  Target walking velocity: {avg_velocity:.3f} m/s")
        if avg_velocity > baseline_vel * 0.7:
            print("✅ Maintains reasonable speed while adding goal-directed behavior!")
        else:
            print("⚠️  Significant speed drop - may need adjustment")
    
    else:
        print("❌ All episodes failed - observation space or other fundamental issue")
    
    env.close()
    return episode_results

if __name__ == "__main__":
    results = test_original_target_walking()
    
    if results and np.mean([r['targets'] for r in results]) >= 1.0:
        print(f"\n🎊 BREAKTHROUGH! Original TargetWalkingWrapper WORKS!")
        print(f"🎯 Strategy: Fine-tune this approach instead of SmoothTargetWrapper!")
    elif results and np.mean([r['velocity'] for r in results]) > 0.15:
        print(f"\n📈 PROMISING! Robot moves well, just needs target tuning!")
    else:
        print(f"\n🤔 Need to try different approach - obs space mismatch likely")