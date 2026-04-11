#!/usr/bin/env python3
"""
Servo Identification Script for RealANT

This script moves each joint individually to help identify:
1. Which servo corresponds to which leg
2. Which direction is "correct" (toe down vs toe up)
3. Whether any servos are inverted

Run this with ant_server.py running to map the physical robot.
"""

import zmq
import time
import sys

def send_command(socket, cmd_string):
    """Send command to robot via ZeroMQ."""
    try:
        socket.send_multipart([b"cmd", cmd_string.encode()])
        return True
    except Exception as e:
        print(f"❌ Error sending command: {e}")
        return False

def test_servo(socket, servo_num, position, duration=2.0):
    """Move a single servo while keeping others at center."""
    # All servos at center (512) except the one we're testing
    positions = [512] * 8
    positions[servo_num - 1] = position

    # Format command
    cmd_parts = [f"s{i+1} {pos}" for i, pos in enumerate(positions)]
    cmd_string = " ".join(cmd_parts) + "\n"

    send_command(socket, cmd_string)
    time.sleep(duration)

def main():
    print("=" * 70)
    print("RealANT Servo Identification Tool")
    print("=" * 70)
    print("\nThis will move each joint individually to help identify:")
    print("  - Which servo controls which leg")
    print("  - Which ankles are inverted (toe up vs toe down)")
    print("\nMake sure:")
    print("  1. ant_server.py is running")
    print("  2. Robot is in a safe position")
    print("  3. You're ready to observe the robot")
    print("\n" + "=" * 70)
    input("Press Enter to start...")

    # Setup ZeroMQ
    print("\n📡 Connecting to robot...")
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:3001")
    time.sleep(1.0)  # Wait for connection
    print("   ✅ Connected!")

    # Attach servos
    print("\n📌 Attaching servos...")
    send_command(socket, "attach_servos\n")
    time.sleep(0.5)

    # Move all to center first
    print("\n🏠 Moving all servos to center position (512)...")
    center_cmd = "s1 512 s2 512 s3 512 s4 512 s5 512 s6 512 s7 512 s8 512\n"
    send_command(socket, center_cmd)
    time.sleep(2.0)
    print("   ✅ All servos at center")

    # Joint information
    joints = [
        ("s1", "hip_1", "Front Left Hip"),
        ("s2", "ankle_1", "Front Left Ankle"),
        ("s3", "hip_2", "Front Right Hip"),
        ("s4", "ankle_2", "Front Right Ankle"),
        ("s5", "hip_3", "Back Left Hip"),
        ("s6", "ankle_3", "Back Left Ankle"),
        ("s7", "hip_4", "Back Right Hip"),
        ("s8", "ankle_4", "Back Right Ankle"),
    ]

    print("\n" + "=" * 70)
    print("TESTING EACH JOINT INDIVIDUALLY")
    print("=" * 70)
    print("\nFor ANKLE joints, observe if the toe points:")
    print("  ✅ DOWN (toward ground) = CORRECT")
    print("  ❌ UP (toward sky) = INVERTED")
    print("\n" + "=" * 70)

    for servo_num, mujoco_name, description in joints:
        print(f"\n{'─' * 70}")
        print(f"Testing: {servo_num} ({mujoco_name}) - {description}")
        print(f"{'─' * 70}")

        # Move to low position (400)
        print(f"  Moving {servo_num} to position 400...")
        test_servo(socket, int(servo_num[1]), 400, duration=1.5)

        # Back to center
        print(f"  Returning {servo_num} to center (512)...")
        test_servo(socket, int(servo_num[1]), 512, duration=1.0)

        # Move to high position (624)
        print(f"  Moving {servo_num} to position 624...")
        test_servo(socket, int(servo_num[1]), 624, duration=1.5)

        # Back to center
        print(f"  Returning {servo_num} to center (512)...")
        test_servo(socket, int(servo_num[1]), 512, duration=1.0)

        # Ask for observation
        if "Ankle" in description:
            print(f"\n❓ Question: When {servo_num} moved, did the toe point:")
            print(f"   A) DOWN (toward ground) = CORRECT ✅")
            print(f"   B) UP (toward sky) = INVERTED ❌")
            response = input(f"   Your answer (A/B): ").strip().upper()

            if response == 'B':
                print(f"   ⚠️  INVERTED: {servo_num} ({mujoco_name}) needs to be flipped!")
            elif response == 'A':
                print(f"   ✅ CORRECT: {servo_num} ({mujoco_name}) is mapped correctly")
            else:
                print(f"   ⚠️  No answer recorded")
        else:
            input(f"   Press Enter to continue to next joint...")

    # Return to center and detach
    print("\n" + "=" * 70)
    print("🏠 Returning all servos to center...")
    send_command(socket, center_cmd)
    time.sleep(1.5)

    print("\n🔌 Detaching servos...")
    send_command(socket, "detach_servos\n")
    time.sleep(0.5)

    print("\n" + "=" * 70)
    print("✅ Servo identification complete!")
    print("=" * 70)
    print("\nBased on your observations, update the calibration with:")
    print("  - Inversion flags for any ankles that point UP")
    print("  - This will flip their control direction")
    print("=" * 70)

    socket.close()
    context.term()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
