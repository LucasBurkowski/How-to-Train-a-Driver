"""
Path generator for inference / deployment.

After training, the policy outputs normalised velocity commands
[vx_norm, vy_norm, omega_norm].  This module converts those commands
into a sequence of waypoints (Pose2d targets) that a WPILib path
follower (e.g. PathPlannerLib) can execute.

Workflow
--------
1. Load the trained PPO model.
2. For each robot, provide the current observation.
3. Call `PathGenerator.generate_path()` to get a list of Pose2d waypoints.
4. Send those waypoints to the path follower on the robot.

Waypoints are spaced `lookahead_dt` seconds apart and projected forward
from the current pose using the commanded velocity, so they represent
where the robot will be if it follows the command faithfully.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from stable_baselines3 import PPO

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Pose2d:
    """A 2-D robot pose (metres, radians)."""
    x: float
    y: float
    theta: float  # radians, field-relative

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "theta_rad": self.theta,
                "theta_deg": math.degrees(self.theta)}


@dataclass
class RobotPath:
    """
    A time-stamped sequence of Pose2d waypoints for one robot.

    dt_seconds:  time between consecutive waypoints
    waypoints:   ordered list of poses (index 0 = current + dt, etc.)
    """
    robot_id: int
    dt_seconds: float
    waypoints: List[Pose2d]

    def __len__(self) -> int:
        return len(self.waypoints)


# ---------------------------------------------------------------------------
# PathGenerator
# ---------------------------------------------------------------------------

class PathGenerator:
    """
    Convert policy actions into path-follower waypoints.

    Parameters
    ----------
    model_path : str
        Path to a StableBaselines3 PPO .zip model file.
    n_waypoints : int
        Number of lookahead waypoints to generate per call.
    lookahead_dt : float
        Time step between consecutive waypoints in seconds.
    max_speed : float
        Maximum translational speed (m/s) – matches robot sim setting.
    max_omega : float
        Maximum rotational speed (rad/s).
    """

    MAX_SPEED_DEFAULT = 4.5
    MAX_OMEGA_DEFAULT = 2 * math.pi

    def __init__(
        self,
        model_path: str,
        n_waypoints: int = 10,
        lookahead_dt: float = 0.1,
        max_speed: float = MAX_SPEED_DEFAULT,
        max_omega: float = MAX_OMEGA_DEFAULT,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 is required for PathGenerator.")

        self.model = PPO.load(model_path, device="cpu")
        self.n_waypoints = n_waypoints
        self.lookahead_dt = lookahead_dt
        self.max_speed = max_speed
        self.max_omega = max_omega

    def generate_path(
        self,
        robot_id: int,
        observation: np.ndarray,
        current_x: float,
        current_y: float,
        current_theta: float,
    ) -> RobotPath:
        """
        Generate a path for one robot given its current observation.

        Parameters
        ----------
        robot_id : int
        observation : np.ndarray of shape (OBS_DIM,)
            The per-robot observation vector.
        current_x, current_y, current_theta : float
            Current robot pose in field coordinates.

        Returns
        -------
        RobotPath
        """
        action, _ = self.model.predict(observation, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        vx = float(action[0]) * self.max_speed
        vy = float(action[1]) * self.max_speed
        omega = float(action[2]) * self.max_omega

        waypoints: List[Pose2d] = []
        x, y, theta = current_x, current_y, current_theta

        for _ in range(self.n_waypoints):
            x += vx * self.lookahead_dt
            y += vy * self.lookahead_dt
            theta = _wrap_angle(theta + omega * self.lookahead_dt)
            waypoints.append(Pose2d(x=x, y=y, theta=theta))

        return RobotPath(
            robot_id=robot_id,
            dt_seconds=self.lookahead_dt,
            waypoints=waypoints,
        )

    def generate_all_paths(
        self,
        joint_observation: np.ndarray,
        current_poses: List[tuple],
    ) -> List[RobotPath]:
        """
        Generate paths for all robots given the joint observation.

        Parameters
        ----------
        joint_observation : np.ndarray of shape (NUM_ROBOTS * OBS_DIM,)
        current_poses : list of (x, y, theta) tuples, one per robot

        Returns
        -------
        List of RobotPath, one per robot
        """
        from frc_rl.env.frc_env import NUM_ROBOTS, OBS_DIM

        paths: List[RobotPath] = []
        for i in range(NUM_ROBOTS):
            obs = joint_observation[i * OBS_DIM : (i + 1) * OBS_DIM]
            x, y, theta = current_poses[i]
            paths.append(self.generate_path(i, obs, x, y, theta))
        return paths


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)
