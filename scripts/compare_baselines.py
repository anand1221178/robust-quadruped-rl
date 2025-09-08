#!/usr/bin/env python3
"""
Compare the baseline in done/ vs TargetWalkingWrapper performance
Make sure we're not losing speed with the wrapper
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.target_walking_wrapper import TargetWalkingWrapper
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def compare_baseline_speeds():
    """Compare raw baseline vs TargetWalkingWrapper speeds"""
    
    print("🏁 BASELINE SPEED COMPARISON")
    print("Raw baseline vs TargetWalkingWrapper")
    print("=" * 50)
    
    model_path = 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip'
    norm_path = 'done/ppo_baseline_ueqbjf2x/vec_normalize.pkl'
    
    results = {}
    
    # Test 1: Raw baseline (how it was trained)
    print("\n🎯 TEST 1: RAW BASELINE (original training setup)")
    print("-" * 45)
    
    def make_baseline_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)  # This is how it was trained
        return env
    
    env1 = DummyVecEnv([make_baseline_env])
    env1 = VecNormalize.load(norm_path, env1)
    env1.training = False
    
    model = PPO.load(model_path)
    
    obs = env1.reset()
    initial_x = env1.envs[0].unwrapped.data.qpos[0]
    
    print("Running raw baseline for 500 steps...")
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env1.step(action)
        
        if (step + 1) % 100 == 0:
            current_x = env1.envs[0].unwrapped.data.qpos[0]
            distance = current_x - initial_x
            velocity = distance / ((step + 1) * 0.05)
            print(f"  Step {step+1:3d}: x={current_x:.3f}m, vel={velocity:.3f}m/s")
        
        if done[0]:
            break
    
    final_x = env1.envs[0].unwrapped.data.qpos[0]
    baseline_distance = final_x - initial_x
    baseline_velocity = baseline_distance / (500 * 0.05)
    
    results['baseline'] = baseline_velocity
    print(f"📊 Raw baseline velocity: {baseline_velocity:.6f} m/s")
    
    env1.close()
    
    # Test 2: With TargetWalkingWrapper
    print(f"\n🎯 TEST 2: WITH TARGETWALKINGWRAPPER")
    print("-" * 45)
    
    def make_target_env():
        env = gym.make('RealAntMujoco-v0')
        env = TargetWalkingWrapper(env, target_distance=5.0)
        return env
    
    env2 = DummyVecEnv([make_target_env])
    env2 = VecNormalize.load(norm_path, env2)
    env2.training = False
    
    obs = env2.reset()
    initial_x = env2.envs[0].unwrapped.data.qpos[0]
    
    print("Running with TargetWalkingWrapper for 500 steps...")
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env2.step(action)
        
        if (step + 1) % 100 == 0:
            current_x = env2.envs[0].unwrapped.data.qpos[0]
            distance = current_x - initial_x
            velocity = distance / ((step + 1) * 0.05)
            print(f"  Step {step+1:3d}: x={current_x:.3f}m, vel={velocity:.3f}m/s")
        
        if done[0]:
            break
    
    final_x = env2.envs[0].unwrapped.data.qpos[0]
    target_distance = final_x - initial_x
    target_velocity = target_distance / (500 * 0.05)
    
    results['target'] = target_velocity
    print(f"📊 TargetWalkingWrapper velocity: {target_velocity:.6f} m/s")
    
    env2.close()
    
    # Comparison
    print(f"\n📊 SPEED COMPARISON:")
    print("=" * 50)
    print(f"Raw baseline velocity:        {results['baseline']:.6f} m/s")
    print(f"TargetWalkingWrapper velocity: {results['target']:.6f} m/s")
    
    speed_ratio = results['target'] / results['baseline']
    speed_diff = results['target'] - results['baseline']
    
    print(f"Difference:                   {speed_diff:+.6f} m/s")
    print(f"Ratio:                        {speed_ratio:.3f} ({speed_ratio*100:.1f}%)")
    
    if abs(speed_diff) < 0.01:
        print("✅ SPEEDS ARE ESSENTIALLY IDENTICAL!")
    elif speed_diff > 0:
        print(f"✅ TargetWalkingWrapper is {speed_diff:.3f} m/s FASTER!")
    else:
        print(f"⚠️  TargetWalkingWrapper is {abs(speed_diff):.3f} m/s slower")
    
    # Final assessment
    print(f"\n🎯 CONCLUSION:")
    if results['target'] > 0.2:
        print("✅ TargetWalkingWrapper maintains good walking speed")
        print("🚀 Ready for robustness training!")
    else:
        print("⚠️  Walking speed might be too low")
        print("🔧 May need adjustment before robustness training")
    
    return results

if __name__ == "__main__":
    results = compare_baseline_speeds()
    print(f"\nFinal speeds: Baseline={results['baseline']:.3f} m/s, Target={results['target']:.3f} m/s")