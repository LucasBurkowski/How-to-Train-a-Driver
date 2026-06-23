"""
WPILib / maple-sim bridge via NetworkTables 4.

This module allows the Python RL environment to communicate with a live
WPILib robot simulation process (running maple-sim) over NT4.

Usage
-----
In standalone (no WPILib process) mode the bridge is a no-op and the
environment relies entirely on the Python RobotSimulator.

When a WPILib simulation is running:
  1. Instantiate WPILibBridge(server_ip="localhost") and call connect().
  2. The bridge publishes desired robot velocities and reads back
     actual robot states from the simulation.

NT4 topic layout (one per robot, indexed by robot_id 0..5):
  /frc_rl/robot/{id}/cmd/vx       (double, commanded vx in m/s)
  /frc_rl/robot/{id}/cmd/vy       (double, commanded vy in m/s)
  /frc_rl/robot/{id}/cmd/omega    (double, commanded omega in rad/s)
  /frc_rl/robot/{id}/state/x      (double)
  /frc_rl/robot/{id}/state/y      (double)
  /frc_rl/robot/{id}/state/theta  (double)
  /frc_rl/robot/{id}/state/vx     (double)
  /frc_rl/robot/{id}/state/vy     (double)
  /frc_rl/robot/{id}/state/omega  (double)
  /frc_rl/robot/{id}/state/has_piece  (boolean)

  /frc_rl/match/reset             (boolean, pulse to reset simulation)
  /frc_rl/match/time_remaining    (double, seconds)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# NT4 is provided by the `ntcore` Python package (part of WPILib).
# We perform a soft import so the rest of the code works without it.
try:
    import ntcore  # type: ignore  # provided by robotpy-ntcore (pip install robotpy-ntcore)

    _NT_AVAILABLE = True
except ImportError:
    _NT_AVAILABLE = False
    logger.warning(
        "robotpy-ntcore not found – WPILibBridge will run in standalone (no-op) mode. "
        "Install with: pip install 'frc-rl-driver[wpilib]'"
    )


class WPILibBridge:
    """
    Two-way NT4 bridge between the Python RL environment and WPILib maple-sim.

    Parameters
    ----------
    server_ip:
        IP of the NT4 server (WPILib simulation host).  Use "localhost" when
        simulation and training run on the same machine.
    num_robots:
        Total number of robots (default 6 for two full alliances).
    connect_timeout:
        Seconds to wait for NT4 connection before raising RuntimeError.
    """

    _NT_PORT = 5810

    def __init__(
        self,
        server_ip: str = "localhost",
        num_robots: int = 6,
        connect_timeout: float = 5.0,
    ) -> None:
        self.server_ip = server_ip
        self.num_robots = num_robots
        self.connect_timeout = connect_timeout
        self._connected = False
        self._inst: Optional[object] = None

        # NT publishers / subscribers (keyed by robot_id)
        self._cmd_pub: Dict[int, Dict[str, object]] = {}
        self._state_sub: Dict[int, Dict[str, object]] = {}
        self._reset_pub: Optional[object] = None
        self._time_sub: Optional[object] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Start NT4 client and wait for the server to respond.

        Returns True if connected, False if NT4 is unavailable or timed out.
        """
        if not _NT_AVAILABLE:
            logger.info("NT4 not available; running in standalone simulation mode.")
            return False

        self._inst = ntcore.NetworkTableInstance.getDefault()
        self._inst.startClient4("frc_rl_trainer")
        self._inst.setServer(self.server_ip, self._NT_PORT)

        deadline = time.monotonic() + self.connect_timeout
        while time.monotonic() < deadline:
            if self._inst.isConnected():
                self._connected = True
                self._setup_topics()
                logger.info("Connected to WPILib simulation at %s", self.server_ip)
                return True
            time.sleep(0.1)

        logger.warning(
            "Timed out waiting for WPILib sim at %s:%d – standalone mode.",
            self.server_ip,
            self._NT_PORT,
        )
        return False

    def disconnect(self) -> None:
        if self._inst is not None and _NT_AVAILABLE:
            self._inst.stopClient()
        self._connected = False

    # ------------------------------------------------------------------
    # NT topic setup
    # ------------------------------------------------------------------

    def _setup_topics(self) -> None:
        if not _NT_AVAILABLE or self._inst is None:
            return

        table = self._inst.getTable("frc_rl")

        # Match-level topics
        self._reset_pub = table.getSubTable("match").getDoubleTopic("reset").publish()
        self._time_sub = table.getSubTable("match").getDoubleTopic("time_remaining").subscribe(0.0)

        for rid in range(self.num_robots):
            robot_table = table.getSubTable(f"robot/{rid}")

            # Command publishers
            cmd_table = robot_table.getSubTable("cmd")
            self._cmd_pub[rid] = {
                "vx": cmd_table.getDoubleTopic("vx").publish(),
                "vy": cmd_table.getDoubleTopic("vy").publish(),
                "omega": cmd_table.getDoubleTopic("omega").publish(),
            }

            # State subscribers
            state_table = robot_table.getSubTable("state")
            self._state_sub[rid] = {
                "x": state_table.getDoubleTopic("x").subscribe(0.0),
                "y": state_table.getDoubleTopic("y").subscribe(0.0),
                "theta": state_table.getDoubleTopic("theta").subscribe(0.0),
                "vx": state_table.getDoubleTopic("vx").subscribe(0.0),
                "vy": state_table.getDoubleTopic("vy").subscribe(0.0),
                "omega": state_table.getDoubleTopic("omega").subscribe(0.0),
                "has_piece": state_table.getBooleanTopic("has_piece").subscribe(False),
            }

    # ------------------------------------------------------------------
    # Runtime I/O
    # ------------------------------------------------------------------

    def send_command(
        self, robot_id: int, vx: float, vy: float, omega: float
    ) -> None:
        """Publish velocity command for one robot."""
        if not self._connected:
            return
        pub = self._cmd_pub.get(robot_id)
        if pub is None:
            return
        pub["vx"].set(vx)
        pub["vy"].set(vy)
        pub["omega"].set(omega)

    def get_robot_state(self, robot_id: int) -> Optional[Dict[str, float]]:
        """
        Read current robot state from NT4.

        Returns None if not connected or the topic has not been published yet.
        """
        if not self._connected:
            return None
        sub = self._state_sub.get(robot_id)
        if sub is None:
            return None
        return {
            "x": sub["x"].get(),
            "y": sub["y"].get(),
            "theta": sub["theta"].get(),
            "vx": sub["vx"].get(),
            "vy": sub["vy"].get(),
            "omega": sub["omega"].get(),
            "has_piece": float(sub["has_piece"].get()),
        }

    def get_time_remaining(self) -> float:
        """Return seconds remaining in the simulated match."""
        if not self._connected or self._time_sub is None:
            return 0.0
        return self._time_sub.get()

    def reset_simulation(self) -> None:
        """Pulse the reset topic to restart the WPILib simulation."""
        if not self._connected or self._reset_pub is None:
            return
        self._reset_pub.set(1.0)
        time.sleep(0.05)
        self._reset_pub.set(0.0)

    @property
    def is_connected(self) -> bool:
        return self._connected
