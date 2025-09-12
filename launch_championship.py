#!/usr/bin/env python3
"""
🚀 Championship Suite Launcher
Quick access to the Ultimate Robustness Championship Suite
"""

import sys
import os

def main():
    print("🏆 Championship Suite Launcher")
    print("=" * 40)
    
    try:
        # Add scripts to path and launch
        sys.path.insert(0, 'scripts')
        from ultimate_championship_suite import main as launch_suite
        launch_suite()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTry running directly:")
        print("python scripts/ultimate_championship_suite.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()