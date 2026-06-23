#!/usr/bin/env python3
"""
Example: Using the field visualizer with robot simulators.

This example shows how to integrate the FieldVisualizer into your
simulation code to visualize robots in real-time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import math
from frc_rl.sim.robot_sim import RobotSimulator
from frc_rl.sim.field_visualizer import FieldVisualizer


def main():
    """Example: Animate 6 robots (3 red, 3 blue) with the visualizer."""

    FIELD_LENGTH = 16.54
    FIELD_WIDTH = 8.21

    # Initialize 3 red robots and 3 blue robots
    red_robots = [
        RobotSimulator(robot_id=0, alliance="red", start_x=2.0, start_y=2.0, start_theta=0.0),
        RobotSimulator(robot_id=1, alliance="red", start_x=2.0, start_y=4.1, start_theta=0.0),
        RobotSimulator(robot_id=2, alliance="red", start_x=2.0, start_y=6.2, start_theta=0.0),
    ]

    blue_robots = [
        RobotSimulator(robot_id=3, alliance="blue", start_x=14.5, start_y=2.0, start_theta=math.pi),
        RobotSimulator(robot_id=4, alliance="blue", start_x=14.5, start_y=4.1, start_theta=math.pi),
        RobotSimulator(robot_id=5, alliance="blue", start_x=14.5, start_y=6.2, start_theta=math.pi),
    ]

    all_robots = red_robots + blue_robots

    # Create the visualizer
    viz = FieldVisualizer(title="Robot Simulator Example")

    # Simulation parameters
    dt = 0.02  # 20ms per step
    num_steps = 1000

    # Run simulation
    for step in range(num_steps):
        # Example: Move robots with simple sinusoidal patterns
        for i, robot in enumerate(all_robots):
            # Red robots move forward, blue robots move backward
            direction = 1 if robot.state.alliance == "red" else -1

            # Desired velocity (normalized to [-1, 1])
            vx_norm = 0.5 * direction * math.sin(step / 100.0)
            vy_norm = 0.3 * math.cos(step / 150.0)
            omega_norm = 0.1 * math.sin(step / 75.0)

            # Step the robot physics
            robot.step(vx_norm, vy_norm, omega_norm, FIELD_LENGTH, FIELD_WIDTH)

        # Collect all robot states
        states = [robot.state for robot in all_robots]

        # Define game piece locations (example)
        game_pieces = [
            (8.27, 2.0),    # Center reef
            (8.27, 8.21),   # Opposite reef
            (8.27, 4.1),    # Middle
        ]

        # Render the frame (this updates the visualization)
        viz.render(states, step=step, game_pieces=game_pieces)

        # Simulate some robots picking up/dropping game pieces
        if step % 150 == 0:
            robot_idx = (step // 150) % len(all_robots)
            all_robots[robot_idx].state.has_game_piece = not all_robots[robot_idx].state.has_game_piece

    print(f"Simulation complete! Ran {num_steps} steps.")
    viz.close()


if __name__ == "__main__":
    main()
