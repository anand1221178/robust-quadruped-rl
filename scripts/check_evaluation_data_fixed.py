#!/usr/bin/env python3
"""
Check evaluation results - handle dictionary data
"""

import numpy as np
import os

def print_nested_dict(data, prefix="", max_depth=3, current_depth=0):
    """Print nested dictionary contents"""
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, np.ndarray)):
                print(f"{prefix}{key}:")
                if isinstance(value, dict):
                    print_nested_dict(value, prefix + "  ", max_depth, current_depth + 1)
                else:
                    # NumPy array
                    if value.shape == ():
                        print(f"{prefix}  {float(value):.6f}")
                    elif len(value.shape) == 1 and len(value) <= 5:
                        print(f"{prefix}  {value}")
                    else:
                        print(f"{prefix}  Shape: {value.shape}, Mean: {np.mean(value):.6f}")
            else:
                print(f"{prefix}{key}: {value}")
    else:
        print(f"{prefix}{data}")

def check_evaluation_results():
    """Load and examine the evaluation results"""
    
    results_path = "sr2l_evaluation_results/sr2l_evaluation_results.npz"
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return
    
    print(f"📊 Loading evaluation results from {results_path}")
    
    try:
        data = np.load(results_path, allow_pickle=True)
        
        print(f"📁 File contains {len(data.files)} items:")
        
        for key in data.files:
            print(f"\n{'='*60}")
            print(f"📊 {key.upper()}:")
            print('='*60)
            
            value = data[key]
            
            if hasattr(value, 'item'):
                # Scalar numpy object that might contain nested data
                actual_value = value.item()
                print_nested_dict(actual_value)
            else:
                print_nested_dict(value)
        
        print("\n" + "🔍" + "="*59)
        print("SEARCHING FOR 1.31 m/s MEASUREMENT...")
        print("="*60)
        
        def search_for_value(obj, path="", target_min=1.25, target_max=1.35):
            """Recursively search for values around 1.31"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    search_for_value(value, f"{path}.{key}" if path else key, target_min, target_max)
            elif isinstance(obj, (list, tuple)):
                for i, value in enumerate(obj):
                    search_for_value(value, f"{path}[{i}]", target_min, target_max)
            elif isinstance(obj, np.ndarray):
                if obj.shape == ():
                    # Scalar
                    val = float(obj)
                    if target_min <= val <= target_max:
                        print(f"🎯 FOUND: {path} = {val:.6f}")
                elif len(obj.shape) == 1:
                    # 1D array
                    mean_val = np.mean(obj)
                    if target_min <= mean_val <= target_max:
                        print(f"🎯 FOUND: {path} mean = {mean_val:.6f} ± {np.std(obj):.6f}")
                    matches = obj[(obj >= target_min) & (obj <= target_max)]
                    if len(matches) > 0:
                        print(f"🎯 FOUND: {path} contains {len(matches)} values around 1.31")
            elif isinstance(obj, (int, float)):
                if target_min <= obj <= target_max:
                    print(f"🎯 FOUND: {path} = {obj:.6f}")
        
        for key in data.files:
            value = data[key]
            if hasattr(value, 'item'):
                search_for_value(value.item(), key)
            else:
                search_for_value(value, key)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_evaluation_results()