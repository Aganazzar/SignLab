#!/usr/bin/env python3
"""
Quick start script for real-time sign language recognition
"""
import subprocess
import sys

def main():
    print("=" * 60)
    print("  Real-Time Sign Language Interpreter")
    print("=" * 60)
    print()
    print("Controls:")
    print("  - ESC: Quit")
    print()
    print("Signs recognized: hello, thank_you, yes, no, please,")
    print("                  sorry, i, you, help")
    print()
    print("=" * 60)
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "realtime.realtime_inference"], check=True)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
