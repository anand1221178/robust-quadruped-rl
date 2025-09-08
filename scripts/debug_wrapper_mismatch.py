#!/usr/bin/env python3
"""
Debug the observation space mismatch issue
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.smooth_target_wrapper import SmoothTargetWrapper
import realant_sim

def debug_observation_spaces():
    """Debug what observation spaces we're dealing with"""
    
    print("🔧 DEBUGGING OBSERVATION SPACE MISMATCH")
    print("=" * 60)
    
    # Check raw environment
    print("1️⃣ RAW REALANT:")
    raw_env = gym.make('RealAntMujoco-v0')
    print(f"   Obs space: {raw_env.observation_space}")
    print(f"   Obs shape: {raw_env.observation_space.shape}")
    raw_env.close()
    
    # Check with SmoothTargetWrapper
    print("\n2️⃣ WITH SMOOTHTARGETWRAPPER:")
    wrapped_env = gym.make('RealAntMujoco-v0')
    wrapped_env = SmoothTargetWrapper(wrapped_env, target_distance=5.0)
    print(f"   Obs space: {wrapped_env.observation_space}")
    print(f"   Obs shape: {wrapped_env.observation_space.shape}")
    
    # Test actual observation
    obs, _ = wrapped_env.reset()
    print(f"   Actual obs shape: {obs.shape}")
    print(f"   Last 2 values (direction): {obs[-2:]}")
    wrapped_env.close()
    
    # Check VecNormalize file
    print("\n3️⃣ CHECKING VECNORMALIZE FILE:")
    norm_path = 'experiments/ppo_smooth_baseline_rohl32fn/vec_normalize.pkl'
    
    try:
        # Create dummy env to load VecNormalize
        dummy_env = DummyVecEnv([lambda: SmoothTargetWrapper(gym.make('RealAntMujoco-v0'), 5.0)])
        vec_norm = VecNormalize.load(norm_path, dummy_env)
        
        print(f"   VecNormalize obs space: {vec_norm.observation_space}")
        print(f"   VecNormalize expects: {vec_norm.observation_space.shape}")
        
        # Test reset
        obs = vec_norm.reset()
        print(f"   VecNormalize obs shape: {obs.shape}")
        
        vec_norm.close()
        
    except Exception as e:
        print(f"   ❌ VecNormalize failed: {e}")
    
    # Check model expectations
    print("\n4️⃣ CHECKING MODEL EXPECTATIONS:")
    model_path = 'experiments/ppo_smooth_baseline_rohl32fn/final_model.zip'
    
    try:
        model = PPO.load(model_path)
        policy = model.policy
        
        if hasattr(policy, 'observation_space'):
            print(f"   Model expects obs space: {policy.observation_space}")
        if hasattr(policy, 'features_extractor'):
            print(f"   Features extractor input: {policy.features_extractor}")
        
        # Try to get input shape from the network
        if hasattr(policy, 'mlp_extractor'):
            if hasattr(policy.mlp_extractor, 'policy_net'):
                first_layer = policy.mlp_extractor.policy_net[0]
                if hasattr(first_layer, 'in_features'):
                    print(f"   First layer expects: {first_layer.in_features} features")
        
    except Exception as e:
        print(f"   ❌ Model inspection failed: {e}")

def test_with_correct_setup():
    """Test with the EXACT same setup as training"""
    
    print(f"\n🎯 TESTING WITH CORRECT SETUP:")
    print("-" * 50)
    
    model_path = 'experiments/ppo_smooth_baseline_rohl32fn/final_model.zip'
    norm_path = 'experiments/ppo_smooth_baseline_rohl32fn/vec_normalize.pkl'
    
    # Create EXACT same environment as training
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SmoothTargetWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize
    try:
        env = VecNormalize.load(norm_path, env)
        env.training = False
        print("✅ VecNormalize loaded successfully")
        
        # Load model
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
        
        # Test one episode
        obs = env.reset()
        print(f"📊 Reset obs shape: {obs.shape}")
        
        total_reward = 0
        targets_reached = 0
        positions = []
        
        for step in range(100):  # Just 100 steps to test
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward[0]
            
            # Track position
            current_x = env.envs[0].unwrapped.data.qpos[0]
            positions.append(current_x)
            
            # Check for target reached
            if info[0] and 'target_reached' in info[0] and info[0]['target_reached']:
                targets_reached += 1
                print(f"  🎯 Target reached at step {step}!")
            
            if step % 20 == 0:
                print(f"  Step {step}: reward_sum={total_reward:.1f}, x={current_x:.2f}")
            
            if done[0]:
                break
        
        distance_traveled = positions[-1] - positions[0] if len(positions) >= 2 else 0
        velocity = distance_traveled / (len(positions) * 0.05)
        
        print(f"\n📊 100-STEP TEST RESULTS:")
        print(f"  Total reward: {total_reward:.1f}")
        print(f"  Targets reached: {targets_reached}")
        print(f"  Distance: {distance_traveled:.2f}m")
        print(f"  Velocity: {velocity:.3f} m/s")
        
        # Assessment
        if total_reward > 100:
            print("✅ Model seems to be working!")
        else:
            print("❌ Model still broken - very low rewards")
            
        env.close()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_observation_spaces()
    test_with_correct_setup()