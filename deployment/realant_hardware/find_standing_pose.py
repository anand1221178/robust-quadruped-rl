#!/usr/bin/env python3
"""
Interactive script to find the robot's natural standing pose.

We'll manually adjust each servo to find comfortable standing positions,
then save those as the initial pose for deployment.
"""

import zmq
import time
import sys

def send_command(socket, cmd_string):
    """Send command to robot."""
    try:
        socket.send_multipart([b"cmd", cmd_string.encode()])
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 70)
    print("Find Natural Standing Pose for RealANT")
    print("=" * 70)
    print("\nWe'll find comfortable standing positions for all joints.")
    print("=" * 70)
    input("Press Enter to start...")

    # Setup ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:3001")
    time.sleep(1.0)

    # Attach servos
    print("\n📌 Attaching servos...")
    send_command(socket, "attach_servos\n")
    time.sleep(0.5)

    # Start with all at center (512)
    positions = [512] * 8

    print("\n🏠 Moving to center (512) for all servos...")
    cmd_parts = [f"s{i+1} {pos}" for i, pos in enumerate(positions)]
    cmd_string = " ".join(cmd_parts) + "\n"
    send_command(socket, cmd_string)
    time.sleep(2.0)

    print("\n" + "=" * 70)
    print("SUGGESTED STANDING POSE")
    print("=" * 70)
    print("\nBased on typical quadruped robots:")
    print("  - Hips: 512 (neutral)")
    print("  - Ankles: 650-700 (legs extended, toes down)")
    print("\nLet's try a standing pose:")
    print("=" * 70)

    # Suggested standing pose
    # Hips at center, ankles extended downward
    standing_pose = [
        512,  # s1 hip_1: center
        650,  # s2 ankle_1: extended
        512,  # s3 hip_2: center
        650,  # s4 ankle_2: extended
        512,  # s5 hip_3: center
        650,  # s6 ankle_3: extended
        512,  # s7 hip_4: center
        650   # s8 ankle_4: extended
    ]

    cmd_parts = [f"s{i+1} {pos}" for i, pos in enumerate(standing_pose)]
    cmd_string = " ".join(cmd_parts) + "\n"

    print(f"\nMoving to suggested standing pose...")
    print(f"  Hips (s1,s3,s5,s7): 512")
    print(f"  Ankles (s2,s4,s6,s8): 650")
    send_command(socket, cmd_string)
    time.sleep(2.0)

    print("\n" + "=" * 70)
    print("Does this look like a good standing position?")
    print("  - Body lifted off ground?")
    print("  - Legs stable?")
    print("  - Toes pointing down?")
    print("=" * 70)

    response = input("\nIs this good? (y/n): ").strip().lower()

    if response == 'y':
        print("\n✅ Great! Using this as starting pose:")
        print(f"   {standing_pose}")
        print("\nAdd this to the reset_to_neutral_position() function:")
        print(f"   neutral_positions = {standing_pose}")
    else:
        print("\n" + "=" * 70)
        print("Manual Adjustment")
        print("=" * 70)
        print("\nLet's adjust each servo individually.")
        print("Try different values between 400-700 for ankles.")
        print("=" * 70)

        current_positions = standing_pose.copy()

        while True:
            print(f"\nCurrent positions: {current_positions}")
            print("\nWhich servo to adjust? (1-8, or 'q' to quit)")
            choice = input("Servo number: ").strip()

            if choice.lower() == 'q':
                break

            try:
                servo_num = int(choice)
                if servo_num < 1 or servo_num > 8:
                    print("Invalid servo number!")
                    continue

                new_pos = int(input(f"New position for s{servo_num} (224-800): ").strip())
                new_pos = max(224, min(800, new_pos))

                current_positions[servo_num - 1] = new_pos

                # Send command
                cmd_parts = [f"s{i+1} {pos}" for i, pos in enumerate(current_positions)]
                cmd_string = " ".join(cmd_parts) + "\n"
                send_command(socket, cmd_string)
                time.sleep(0.5)

                print(f"✅ Moved s{servo_num} to {new_pos}")

            except ValueError:
                print("Invalid input!")

        print("\n✅ Final standing pose:")
        print(f"   {current_positions}")
        print("\nUse this in reset_to_neutral_position()!")

    # Detach
    print("\n🔌 Detaching servos...")
    send_command(socket, "detach_servos\n")
    time.sleep(0.5)

    socket.close()
    context.term()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(0)
