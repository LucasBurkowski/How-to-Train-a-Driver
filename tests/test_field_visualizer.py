"""Tests for the field visualizer."""

import math
import sys
from pathlib import Path

import pytest
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frc_rl.sim.robot_sim import RobotState, RobotSimulator
from frc_rl.sim.field_visualizer import FieldVisualizer, visualize_frame


@pytest.fixture
def sample_states():
    """Create sample robot states for testing."""
    return [
        RobotState(robot_id=0, alliance="red", x=2.0, y=2.0, theta=0.0),
        RobotState(robot_id=1, alliance="red", x=2.0, y=4.1, theta=math.pi / 2),
        RobotState(robot_id=3, alliance="blue", x=14.5, y=6.2, theta=math.pi),
    ]


class TestFieldVisualizer:
    """Test the FieldVisualizer class."""

    def test_visualizer_init(self):
        """Test initializing the visualizer."""
        viz = FieldVisualizer()
        assert viz.field_length == 16.54
        assert viz.field_width == 8.21
        viz.close()

    def test_visualizer_custom_dimensions(self):
        """Test visualizer with custom field dimensions."""
        viz = FieldVisualizer(field_length=20.0, field_width=10.0)
        assert viz.field_length == 20.0
        assert viz.field_width == 10.0
        viz.close()

    def test_render_single_frame(self, sample_states):
        """Test rendering a single frame."""
        viz = FieldVisualizer()
        # Should not raise any exception
        viz.render(sample_states, step=0)
        viz.close()

    def test_render_multiple_frames(self, sample_states):
        """Test rendering multiple frames."""
        viz = FieldVisualizer()
        for step in range(5):
            viz.render(sample_states, step=step)
        viz.close()

    def test_render_with_game_pieces(self, sample_states):
        """Test rendering with game pieces."""
        viz = FieldVisualizer()
        game_pieces = [(8.27, 4.1), (8.27, 2.0), (8.27, 6.2)]
        viz.render(sample_states, step=0, game_pieces=game_pieces)
        viz.close()

    def test_render_with_game_piece_status(self, sample_states):
        """Test rendering robots with game pieces."""
        viz = FieldVisualizer()
        sample_states[0].has_game_piece = True
        sample_states[2].has_game_piece = True
        viz.render(sample_states, step=0)
        viz.close()

    def test_render_empty_states(self):
        """Test rendering with no robots."""
        viz = FieldVisualizer()
        viz.render([], step=0)
        viz.close()

    def test_visualizer_cleanup(self):
        """Test that visualizer closes cleanly."""
        viz = FieldVisualizer()
        viz.close()
        # Should be able to create another instance after closing
        viz2 = FieldVisualizer()
        viz2.close()


class TestVisualizeFunctionFrame:
    """Test the visualize_frame function."""

    def test_visualize_frame_basic(self, sample_states):
        """Test one-shot visualization."""
        # Should not raise any exception
        visualize_frame(sample_states, step=0)
        plt.close("all")

    def test_visualize_frame_with_pieces(self, sample_states):
        """Test one-shot visualization with game pieces."""
        game_pieces = [(8.27, 4.1), (8.27, 2.0)]
        visualize_frame(sample_states, step=5, game_pieces=game_pieces)
        plt.close("all")

    def test_visualize_frame_custom_dimensions(self, sample_states):
        """Test one-shot visualization with custom field dimensions."""
        visualize_frame(
            sample_states,
            field_length=20.0,
            field_width=10.0,
            step=0,
        )
        plt.close("all")

    def test_visualize_frame_empty(self):
        """Test one-shot visualization with no robots."""
        visualize_frame([], step=0)
        plt.close("all")


class TestRobotSimulationIntegration:
    """Test visualizer with actual robot simulators."""

    def test_visualize_moving_robots(self):
        """Test visualizing robots moving."""
        # Create robots
        robots = [
            RobotSimulator(0, "red", 2.0, 2.0, 0.0),
            RobotSimulator(3, "blue", 14.5, 6.2, math.pi),
        ]

        viz = FieldVisualizer()

        # Simulate a few steps
        for step in range(10):
            for robot in robots:
                robot.step(0.3, 0.1, 0.05, 16.54, 8.21)

            states = [r.state for r in robots]
            viz.render(states, step=step)

        viz.close()

    def test_wall_collision_visualization(self):
        """Test visualizing robot hitting walls."""
        robot = RobotSimulator(0, "red", 0.5, 4.1, 0.0)
        viz = FieldVisualizer()

        # Drive robot into wall
        for step in range(20):
            robot.step(-1.0, 0.0, 0.0, 16.54, 8.21)  # Drive left
            viz.render([robot.state], step=step)

        viz.close()


class TestRobotStateVisualization:
    """Test visualizing different robot states."""

    def test_all_alliances(self):
        """Test visualizing both alliances."""
        states = []
        for i in range(3):
            states.append(RobotState(
                robot_id=i,
                alliance="red",
                x=2.0 + i,
                y=2.0,
                theta=0.0,
            ))
        for i in range(3):
            states.append(RobotState(
                robot_id=3 + i,
                alliance="blue",
                x=14.5 - i,
                y=6.2,
                theta=math.pi,
            ))

        visualize_frame(states, step=0)
        plt.close("all")

    def test_various_orientations(self):
        """Test visualizing robots with different headings."""
        states = []
        for i in range(4):
            angle = (i / 4) * 2 * math.pi
            states.append(RobotState(
                robot_id=i,
                alliance="red",
                x=4.0 + i * 2,
                y=4.1,
                theta=angle,
            ))

        visualize_frame(states, step=0)
        plt.close("all")

    def test_game_piece_possession(self):
        """Test visualizing robots with/without game pieces."""
        states = []
        for i in range(3):
            states.append(RobotState(
                robot_id=i,
                alliance="red",
                x=2.0 + i,
                y=2.0,
                theta=0.0,
                has_game_piece=(i % 2 == 0),
            ))

        visualize_frame(states, step=0)
        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
