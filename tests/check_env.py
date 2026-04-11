#!/usr/bin/env python3
"""
Test that Phase 1 setup is complete and ready for PPO implementation
"""

import os
import yaml
import gymnasium as gym
import warnings

# Suppress Ant-v4 deprecation warning
warnings.filterwarnings("ignore", message=".*The environment Ant-v4 is out of date.*")

def check_project_structure():
    """Check if project directories exist"""
    print("=== Checking Project Structure ===")
    
    required_dirs = [
        "configs",
        "configs/experiments", 
        "configs/env",
        "configs/train",
        "src",
        "src/agents",
        "src/envs",
        "src/utils",
        "experiments",
        "scripts"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_config_files():
    """Check if configuration files exist and are valid"""
    print("\n=== Checking Configuration Files ===")
    
    configs_to_check = {
        "configs/train/default.yaml": ["ppo", "policy", "seed"],
        "configs/env/realant.yaml": ["env", "env.name"],
        "configs/experiments/ppo_baseline.yaml": ["experiment", "defaults"]
    }
    
    all_valid = True
    
    for config_path, required_keys in configs_to_check.items():
        if os.path.exists(config_path):
            print(f"\n✓ Found {config_path}")
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check required keys
                for key in required_keys:
                    if '.' in key:  # Nested key
                        keys = key.split('.')
                        value = config
                        for k in keys:
                            value = value.get(k, None)
                            if value is None:
                                break
                    else:
                        value = config.get(key)
                    
                    if value is not None:
                        print(f"  ✓ {key}: {value}")
                    else:
                        print(f"  ✗ {key}: MISSING")
                        all_valid = False
                        
            except Exception as e:
                print(f"  ✗ Error reading file: {e}")
                all_valid = False
        else:
            print(f"✗ {config_path} - MISSING")
            all_valid = False
    
    return all_valid

def check_environment():
    """Check if Ant-v4 environment works"""
    print("\n=== Checking Ant-v4 Environment ===")
    
    try:
        env = gym.make('Ant-v4')
        print("✓ Can create Ant-v4 environment")
        
        obs, info = env.reset()
        print(f"✓ Reset works - observation shape: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step works - reward: {reward:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n=== Checking Dependencies ===")
    
    dependencies = {
        "gymnasium": "0.28.0",
        "stable_baselines3": "2.0.0",
        "torch": None,  # Version doesn't matter much
        "numpy": "1.24.0",
        "mujoco": None,
        "wandb": None,
        "tensorboard": None,
        "matplotlib": None,
        "pandas": None,
        "yaml": None
    }
    
    all_installed = True
    
    for package, min_version in dependencies.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package} ({version})")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def main():
    print("=" * 60)
    print("Phase 1 Setup Verification")
    print("=" * 60)
    
    checks = {
        "Project Structure": check_project_structure(),
        "Config Files": check_config_files(),
        "Environment": check_environment(),
        "Dependencies": check_dependencies()
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    if all_passed:
        print("\n✅ Phase 1 setup is complete! Ready to implement PPO baseline.")
        print("\nNext steps:")
        print("1. Create src/train.py for the main training loop")
        print("2. Implement basic PPO training")
        print("3. Design reward function for forward locomotion")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Run: python setup_project.py")
        print("- Update configs/env/realant.yaml to use 'Ant-v4'")
        print("- Install missing packages: pip install -r requirements.txt")

if __name__ == "__main__":
    main()