#!/usr/bin/env python3
"""
Standalone field visualization viewer.

This script can be used to visualize robot states from any simulation
in real-time or replay mode.

Usage
-----
Import and create a viewer, then call render() with robot states:

    from frc_rl.sim.field_visualizer import FieldVisualizer

    viz = FieldVisualizer()
    for step in range(1000):
        # ... run simulation step ...
        viz.render(robot_states, step=step)
    viz.close()

For batch visualization of recorded states, see visualize_recording().
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frc_rl.sim.field_visualizer import FieldVisualizer


def interactive_demo():
    """Launch an interactive demo of the visualizer."""
    print("Starting interactive field visualizer demo...")
    print("This demo will show 6 robots (3 red, 3 blue) moving around the field.")
    print("Close the window to exit.")

    # Import here to allow running without full environment setup initially
    import math
    from frc_rl.sim.robot_sim import RobotSimulator

    FIELD_LENGTH = 16.54
    FIELD_WIDTH = 8.21

    # Create robots
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

    # Create visualizer
    viz = FieldVisualizer(title="FRC Robot Simulation - Interactive Demo")

    # Simulate and visualize
    try:
        for step in range(500):
            # Simple movement pattern
            for i, robot in enumerate(all_robots):
                vx_norm = 0.5 * math.sin(step / 50.0)
                vy_norm = 0.3 * math.cos(step / 100.0)
                omega_norm = 0.1 * math.sin(step / 75.0)

                robot.step(vx_norm, vy_norm, omega_norm, FIELD_LENGTH, FIELD_WIDTH)

            # Collect states
            states = [robot.state for robot in all_robots]

            # Add some game pieces
            game_pieces = [
                (8.27, 2.0),
                (8.27, 8.21),
                (8.27, 4.1),
            ]

            # Render
            viz.render(states, step=step, game_pieces=game_pieces)

            # Update some game piece states
            if step % 100 == 0:
                all_robots[step % 6].state.has_game_piece = not all_robots[step % 6].state.has_game_piece

        print("Demo completed!")
    finally:
        viz.close()


if __name__ == "__main__":
    interactive_demo()
