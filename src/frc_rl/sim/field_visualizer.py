"""
Matplotlib-based GUI for visualizing simulated robots on the FRC field.

Displays robot positions, orientations, and game piece state in real-time
during standalone Python simulation (without WPILib connection).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from frc_rl.sim.robot_sim import RobotState


# FRC field dimensions (Reefscape 2025)
FIELD_LENGTH = 16.54  # metres
FIELD_WIDTH = 8.21    # metres

# Alliance colors
ALLIANCE_COLORS = {
    "red": "#e41e3f",
    "blue": "#003bb6",
}

# Robot styling
ROBOT_SIZE = 0.42  # metres (half the bumper width for visualization)
ROBOT_OUTLINE_WIDTH = 2
TEXT_SIZE = 8


class FieldVisualizer:
    """Real-time field visualization using Matplotlib."""

    def __init__(
        self,
        field_length: float = FIELD_LENGTH,
        field_width: float = FIELD_WIDTH,
        title: str = "FRC Robot Simulation",
    ) -> None:
        """
        Initialize the field visualizer.

        Parameters
        ----------
        field_length : float
            Field length in metres.
        field_width : float
            Field width in metres.
        title : str
            Window title.
        """
        self.field_length = field_length
        self.field_width = field_width

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.suptitle(title, fontsize=14, fontweight="bold")

        # Configure axis
        self.ax.set_xlim(-0.5, field_length + 0.5)
        self.ax.set_ylim(-0.5, field_width + 0.5)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (metres)", fontsize=10)
        self.ax.set_ylabel("Y (metres)", fontsize=10)
        self.ax.grid(True, alpha=0.3)

        # Draw field boundary
        field_rect = patches.Rectangle(
            (0, 0), field_length, field_width,
            linewidth=2, edgecolor="black", facecolor="none"
        )
        self.ax.add_patch(field_rect)

        # Storage for visual elements
        self.robot_circles: Dict[int, patches.Circle] = {}
        self.robot_arrows: Dict[int, patches.FancyArrow] = {}
        self.robot_texts: Dict[int, plt.Text] = {}
        self.step_counter_text: Optional[plt.Text] = None

        plt.tight_layout()

    def render(
        self,
        robot_states: List[RobotState],
        step: int = 0,
        game_pieces: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Render the current frame with all robots.

        Parameters
        ----------
        robot_states : List[RobotState]
            List of robot states to display.
        step : int
            Current simulation step (for display).
        game_pieces : Optional[List[Tuple[float, float]]]
            List of (x, y) positions of unclaimed game pieces.
        """
        # Clear old artists (but keep the field boundary)
        for circle in self.robot_circles.values():
            circle.remove()
        for arrow in self.robot_arrows.values():
            arrow.remove()
        for text in self.robot_texts.values():
            text.remove()
        if self.step_counter_text:
            self.step_counter_text.remove()

        self.robot_circles.clear()
        self.robot_arrows.clear()
        self.robot_texts.clear()

        # Draw game pieces
        if game_pieces:
            for px, py in game_pieces:
                piece = patches.Circle(
                    (px, py), 0.08, color="gold", alpha=0.7, zorder=2
                )
                self.ax.add_patch(piece)

        # Draw each robot
        for state in robot_states:
            color = ALLIANCE_COLORS.get(state.alliance, "gray")
            robot_id = state.robot_id

            # Draw robot body (circle)
            circle = patches.Circle(
                (state.x, state.y),
                ROBOT_SIZE,
                facecolor=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=ROBOT_OUTLINE_WIDTH,
                zorder=3,
            )
            self.ax.add_patch(circle)
            self.robot_circles[robot_id] = circle

            # Draw heading arrow
            arrow_len = ROBOT_SIZE * 0.8
            dx = arrow_len * math.cos(state.theta)
            dy = arrow_len * math.sin(state.theta)
            arrow = patches.FancyArrow(
                state.x, state.y, dx, dy,
                head_width=ROBOT_SIZE * 0.4,
                head_length=ROBOT_SIZE * 0.3,
                fc="white", ec="black", linewidth=1, zorder=4
            )
            self.ax.add_patch(arrow)
            self.robot_arrows[robot_id] = arrow

            # Draw robot ID and game piece indicator
            label = f"R{robot_id}"
            if state.has_game_piece:
                label += " ◆"
            text = self.ax.text(
                state.x, state.y,
                label,
                fontsize=TEXT_SIZE,
                ha="center", va="center",
                fontweight="bold",
                color="white",
                zorder=5,
            )
            self.robot_texts[robot_id] = text

        # Draw step counter
        self.step_counter_text = self.ax.text(
            0.02, 0.98,
            f"Step: {step}",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Redraw
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # Allow UI to update

    def close(self) -> None:
        """Close the visualization window."""
        plt.close(self.fig)


# ============================================================================
# Convenience function for quick visualization
# ============================================================================

def visualize_frame(
    robot_states: List[RobotState],
    field_length: float = FIELD_LENGTH,
    field_width: float = FIELD_WIDTH,
    step: int = 0,
    game_pieces: Optional[List[Tuple[float, float]]] = None,
) -> None:
    """
    One-shot visualization of a single frame (for debugging/inspection).

    Creates a static matplotlib figure showing the current state.

    Parameters
    ----------
    robot_states : List[RobotState]
        List of robot states to display.
    field_length : float
        Field length in metres.
    field_width : float
        Field width in metres.
    step : int
        Current simulation step (for label).
    game_pieces : Optional[List[Tuple[float, float]]]
        List of (x, y) positions of unclaimed game pieces.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"FRC Field State (Step {step})", fontsize=14, fontweight="bold")

    # Configure axis
    ax.set_xlim(-0.5, field_length + 0.5)
    ax.set_ylim(-0.5, field_width + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (metres)", fontsize=10)
    ax.set_ylabel("Y (metres)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Draw field boundary
    field_rect = patches.Rectangle(
        (0, 0), field_length, field_width,
        linewidth=2, edgecolor="black", facecolor="none"
    )
    ax.add_patch(field_rect)

    # Draw game pieces
    if game_pieces:
        for px, py in game_pieces:
            piece = patches.Circle((px, py), 0.08, color="gold", alpha=0.7, zorder=2)
            ax.add_patch(piece)

    # Draw robots
    for state in robot_states:
        color = ALLIANCE_COLORS.get(state.alliance, "gray")

        # Robot body
        circle = patches.Circle(
            (state.x, state.y),
            ROBOT_SIZE,
            facecolor=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=ROBOT_OUTLINE_WIDTH,
            zorder=3,
        )
        ax.add_patch(circle)

        # Heading arrow
        arrow_len = ROBOT_SIZE * 0.8
        dx = arrow_len * math.cos(state.theta)
        dy = arrow_len * math.sin(state.theta)
        arrow = patches.FancyArrow(
            state.x, state.y, dx, dy,
            head_width=ROBOT_SIZE * 0.4,
            head_length=ROBOT_SIZE * 0.3,
            fc="white", ec="black", linewidth=1, zorder=4
        )
        ax.add_patch(arrow)

        # Label
        label = f"R{state.robot_id}"
        if state.has_game_piece:
            label += " ◆"
        ax.text(
            state.x, state.y,
            label,
            fontsize=TEXT_SIZE,
            ha="center", va="center",
            fontweight="bold",
            color="white",
            zorder=5,
        )

    plt.tight_layout()
    plt.show()
