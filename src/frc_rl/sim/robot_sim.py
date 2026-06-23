"""
Standalone Python robot physics simulator.

Approximates maple-sim swerve-drive kinematics so training can run
entirely in Python without a live WPILib/maple-sim process.

Coordinate system: metres, origin at field corner (0, 0).
Angles in radians, CCW positive.

When a WPILibBridge is connected the simulator delegates state updates
to the bridge and only provides helper methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    """Full kinematic state of one robot."""
    robot_id: int
    alliance: str          # "red" | "blue"
    x: float = 0.0        # metres
    y: float = 0.0        # metres
    theta: float = 0.0    # radians, field-relative heading
    vx: float = 0.0       # m/s, field-relative
    vy: float = 0.0       # m/s, field-relative
    omega: float = 0.0    # rad/s
    has_game_piece: bool = False
    is_climbing: bool = False
    # Scores accumulated by this robot during the match
    scored_pieces: int = 0

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_array(self) -> np.ndarray:
        """Return state as a float32 array [x, y, θ, vx, vy, ω, has_piece]."""
        return np.array(
            [self.x, self.y, self.theta, self.vx, self.vy, self.omega,
             float(self.has_game_piece)],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Physics simulator
# ---------------------------------------------------------------------------

class RobotSimulator:
    """
    Euler-integrated swerve-drive dynamics.

    The robot has:
      - Maximum translational speed  : MAX_SPEED m/s
      - Maximum rotational speed     : MAX_OMEGA rad/s
      - First-order translational lag: TAU_TRANS s
      - First-order rotational lag   : TAU_ROT s

    Actions are *desired* (vx, vy, omega) in field-relative coordinates,
    normalised to [-1, 1].
    """

    MAX_SPEED: float = 4.5    # m/s  (fast FRC swerve)
    MAX_OMEGA: float = 2 * math.pi  # rad/s  (one full rotation/s)
    TAU_TRANS: float = 0.10   # velocity lag time constant (s)
    TAU_ROT: float = 0.08

    BUMPER_HALF: float = 0.419  # half of 0.838 m bumper

    def __init__(
        self,
        robot_id: int,
        alliance: str,
        start_x: float,
        start_y: float,
        start_theta: float,
        dt: float = 0.02,
    ) -> None:
        self.dt = dt
        self.state = RobotState(
            robot_id=robot_id,
            alliance=alliance,
            x=start_x,
            y=start_y,
            theta=start_theta,
        )

    def reset(self, x: float, y: float, theta: float) -> None:
        s = self.state
        s.x, s.y, s.theta = x, y, theta
        s.vx = s.vy = s.omega = 0.0
        s.has_game_piece = False
        s.is_climbing = False
        s.scored_pieces = 0

    def step(
        self,
        desired_vx_norm: float,
        desired_vy_norm: float,
        desired_omega_norm: float,
        field_length: float,
        field_width: float,
    ) -> None:
        """
        Advance robot physics by one timestep.

        Parameters
        ----------
        desired_vx_norm, desired_vy_norm, desired_omega_norm:
            Normalised commands in [-1, 1] (field-relative).
        field_length, field_width:
            Field boundaries for wall-clamp.
        """
        s = self.state
        dt = self.dt

        # Desired velocities (physical units)
        target_vx = float(np.clip(desired_vx_norm, -1, 1)) * self.MAX_SPEED
        target_vy = float(np.clip(desired_vy_norm, -1, 1)) * self.MAX_SPEED
        target_omega = float(np.clip(desired_omega_norm, -1, 1)) * self.MAX_OMEGA

        # First-order lag (simulates motor/drive response)
        alpha_t = dt / (self.TAU_TRANS + dt)
        alpha_r = dt / (self.TAU_ROT + dt)
        s.vx += alpha_t * (target_vx - s.vx)
        s.vy += alpha_t * (target_vy - s.vy)
        s.omega += alpha_r * (target_omega - s.omega)

        # Integrate position
        s.x += s.vx * dt
        s.y += s.vy * dt
        s.theta = _wrap_angle(s.theta + s.omega * dt)

        # Clamp to field (wall bounce = zero velocity in that axis)
        half = self.BUMPER_HALF
        if s.x < half:
            s.x = half
            s.vx = max(s.vx, 0.0)
        elif s.x > field_length - half:
            s.x = field_length - half
            s.vx = min(s.vx, 0.0)

        if s.y < half:
            s.y = half
            s.vy = max(s.vy, 0.0)
        elif s.y > field_width - half:
            s.y = field_width - half
            s.vy = min(s.vy, 0.0)

    # ------------------------------------------------------------------
    # Collision detection between robots
    # ------------------------------------------------------------------

    @staticmethod
    def robots_in_contact(a: RobotState, b: RobotState, threshold: float = 0.84) -> bool:
        """Return True if two robots are close enough to be considered in contact."""
        dist = math.hypot(a.x - b.x, a.y - b.y)
        return dist < threshold

    # ------------------------------------------------------------------
    # Wall contact check
    # ------------------------------------------------------------------

    @classmethod
    def in_wall_contact(
        cls, state: RobotState, field_length: float, field_width: float, tolerance: float = 0.05
    ) -> bool:
        half = cls.BUMPER_HALF + tolerance
        return (
            state.x <= half
            or state.x >= field_length - half
            or state.y <= half
            or state.y >= field_width - half
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((angle + math.pi) % (2 * math.pi) - math.pi)
