#!/usr/bin/env python3
"""
Simple demo of the field visualizer with robot_sim.

This script creates some robots and animates them moving around the field
to demonstrate the visualization capabilities.
"""

import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from frc_rl.sim.robot_sim import RobotSimulator, RobotState
from frc_rl.sim.field_visualizer import FieldVisualizer


FIELD_LENGTH = 16.54
FIELD_WIDTH = 8.21


def run_demo():
    """Run a simple demo animation."""
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
    viz = FieldVisualizer()

    # Simulate and visualize
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

    viz.close()
    print("Demo completed!")


if __name__ == "__main__":
    run_demo()
