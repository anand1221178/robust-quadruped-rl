#!/usr/bin/env python3
"""
Check what's in the sr2l evaluation results that gave the 1.31 m/s measurement
"""

import numpy as np
import os

def check_evaluation_results():
    """Load and examine the evaluation results"""
    
    results_path = "sr2l_evaluation_results/sr2l_evaluation_results.npz"
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return
    
    print(f"📊 Loading evaluation results from {results_path}")
    
    try:
        data = np.load(results_path, allow_pickle=True)
        
        print(f"📁 File contains {len(data.files)} arrays:")
        for key in data.files:
            print(f"  - {key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")
        
        # Print the actual data
        print("\n📈 EVALUATION RESULTS:")
        print("=" * 50)
        
        for key in sorted(data.files):
            value = data[key]
            print(f"\n{key}:")
            if hasattr(value, 'shape') and value.shape == ():
                # Scalar value
                print(f"  {float(value):.6f}")
            elif hasattr(value, 'shape') and len(value.shape) == 1 and len(value) <= 10:
                # Short array
                print(f"  {value}")
            elif hasattr(value, 'shape'):
                # Longer array - show stats
                print(f"  Shape: {value.shape}")
                print(f"  Mean: {np.mean(value):.6f}")
                print(f"  Std:  {np.std(value):.6f}")
                print(f"  Min:  {np.min(value):.6f}")
                print(f"  Max:  {np.max(value):.6f}")
            else:
                print(f"  {value}")
        
        print("\n" + "=" * 50)
        print("🔍 LOOKING FOR 1.31 m/s MEASUREMENT...")
        
        # Look for values around 1.31
        for key in data.files:
            value = data[key]
            if hasattr(value, 'shape'):
                if value.shape == ():
                    # Scalar
                    if 1.25 <= float(value) <= 1.35:
                        print(f"🎯 FOUND MATCH: {key} = {float(value):.6f}")
                else:
                    # Array - check mean
                    mean_val = np.mean(value)
                    if 1.25 <= mean_val <= 1.35:
                        print(f"🎯 FOUND MATCH: {key} mean = {mean_val:.6f}")
                    
                    # Check individual values
                    if len(value.shape) == 1:
                        matches = value[(value >= 1.25) & (value <= 1.35)]
                        if len(matches) > 0:
                            print(f"🎯 FOUND VALUES: {key} contains {len(matches)} values around 1.31: {matches}")
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")

if __name__ == "__main__":
    check_evaluation_results()