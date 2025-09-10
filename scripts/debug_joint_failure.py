#!/usr/bin/env python3
"""
DEBUG: Check if joint failures are actually working
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import argparse

class DebugJointFailureWrapper(gym.Wrapper):
    """Debug wrapper to see what's actually happening with joint failures"""
    
    def __init__(self, env, failed_joints):
        super().__init__(env)
        self.failed_joints = failed_joints
        self.joint_names = ["hip_4", "ankle_4", "hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3"]
        self.step_count = 0
        print(f"🔍 DEBUG: Will fail joints {failed_joints} = {[self.joint_names[i] for i in failed_joints]}")
    
    def step(self, action):
        self.step_count += 1
        
        # Print original action
        if self.step_count % 50 == 1:
            print(f"\nStep {self.step_count}:")
            print(f"  Original action: {action}")
        
        # Force failed joints to 0
        modified_action = action.copy()
        for joint_idx in self.failed_joints:
            if self.step_count % 50 == 1:
                print(f"  Setting joint {joint_idx} ({self.joint_names[joint_idx]}) from {modified_action[joint_idx]:.3f} to 0.0")
            modified_action[joint_idx] = 0.0
        
        if self.step_count % 50 == 1:
            print(f"  Modified action: {modified_action}")
        
        # Call environment
        result = self.env.step(modified_action)
        
        return result

def debug_joint_failure(model_folder, failed_joints, steps=200):
    """Debug joint failure to see what's happening"""
    
    model_name = os.path.basename(model_folder)
    joint_names = ["hip_4", "ankle_4", "hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3"]
    
    print(f"🔍 DEBUG: {model_name}")
    print(f"❌ Testing joints: {failed_joints} = {[joint_names[i] for i in failed_joints]}")
    print("=" * 60)
    
    # Model paths
    model_path = os.path.join(model_folder, 'final_model.zip')
    vec_normalize_path = os.path.join(model_folder, 'vec_normalize.pkl')
    
    # Create environment
    env = gym.make('RealAntMujoco-v0')
    env = SuccessRewardWrapper(env)
    env = DebugJointFailureWrapper(env, failed_joints)
    
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False
    
    # Load model  
    model = PPO.load(model_path)
    
    print(f"📊 Action space: {env.action_space}")
    print(f"🎯 Running {steps} steps to check joint failure...")
    
    # Test
    obs = env.reset()
    total_reward = 0
    
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
        if done.any():
            print(f"Episode ended at step {step}")
            break
    
    print(f"\n📊 RESULTS:")
    print(f"  Total reward: {total_reward:.0f}")
    print(f"  Steps completed: {step+1}")
    print(f"\n🔍 Check the debug output above to see if joints are actually being set to 0!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', help='Path to model folder')
    parser.add_argument('--joints', type=int, nargs='+', default=[2], help='Joint indices to fail')
    parser.add_argument('--steps', type=int, default=200, help='Number of steps')
    
    args = parser.parse_args()
    
    debug_joint_failure(args.model_folder, args.joints, args.steps)

if __name__ == "__main__":
    main()