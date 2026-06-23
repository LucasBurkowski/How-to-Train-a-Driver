"""Tests for the robot physics simulator."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frc_rl.env.field import FIELD_LENGTH, FIELD_WIDTH
from frc_rl.sim.robot_sim import RobotSimulator, RobotState, _wrap_angle


@pytest.fixture
def sim() -> RobotSimulator:
    return RobotSimulator(
        robot_id=0, alliance="blue",
        start_x=2.0, start_y=4.0, start_theta=0.0,
        dt=0.02,
    )


class TestRobotSimulator:
    def test_initial_state(self, sim):
        s = sim.state
        assert s.x == pytest.approx(2.0)
        assert s.y == pytest.approx(4.0)
        assert s.theta == pytest.approx(0.0)
        assert s.vx == pytest.approx(0.0)

    def test_step_moves_forward(self, sim):
        for _ in range(50):
            sim.step(1.0, 0.0, 0.0, FIELD_LENGTH, FIELD_WIDTH)
        assert sim.state.x > 2.0

    def test_wall_clamping_x(self, sim):
        # Drive hard into the +X wall
        for _ in range(500):
            sim.step(1.0, 0.0, 0.0, FIELD_LENGTH, FIELD_WIDTH)
        assert sim.state.x <= FIELD_LENGTH

    def test_wall_clamping_y(self, sim):
        for _ in range(500):
            sim.step(0.0, 1.0, 0.0, FIELD_LENGTH, FIELD_WIDTH)
        assert sim.state.y <= FIELD_WIDTH

    def test_reset_clears_velocity(self, sim):
        sim.step(1.0, 1.0, 1.0, FIELD_LENGTH, FIELD_WIDTH)
        sim.reset(3.0, 3.0, 1.0)
        assert sim.state.vx == pytest.approx(0.0)
        assert sim.state.vy == pytest.approx(0.0)

    def test_as_array_length(self, sim):
        arr = sim.state.as_array()
        assert arr.shape == (7,)

    def test_rotation(self, sim):
        for _ in range(100):
            sim.step(0.0, 0.0, 1.0, FIELD_LENGTH, FIELD_WIDTH)
        # Theta should have changed
        assert abs(sim.state.theta) > 0.01

    def test_robots_in_contact_true(self):
        a = RobotState(0, "blue", x=0.0, y=0.0)
        b = RobotState(1, "red", x=0.5, y=0.0)
        assert RobotSimulator.robots_in_contact(a, b, threshold=0.84) is True

    def test_robots_in_contact_false(self):
        a = RobotState(0, "blue", x=0.0, y=0.0)
        b = RobotState(1, "red", x=5.0, y=5.0)
        assert RobotSimulator.robots_in_contact(a, b) is False

    def test_in_wall_contact_at_corner(self):
        s = RobotState(0, "blue", x=0.1, y=4.0)
        assert RobotSimulator.in_wall_contact(s, FIELD_LENGTH, FIELD_WIDTH) is True

    def test_in_wall_contact_inside(self):
        s = RobotState(0, "blue", x=5.0, y=4.0)
        assert RobotSimulator.in_wall_contact(s, FIELD_LENGTH, FIELD_WIDTH) is False


class TestWrapAngle:
    def test_wraps_over_pi(self):
        assert _wrap_angle(4.0) == pytest.approx(4.0 - 2 * math.pi, abs=1e-6)

    def test_wraps_under_minus_pi(self):
        assert _wrap_angle(-4.0) == pytest.approx(-4.0 + 2 * math.pi, abs=1e-6)

    def test_identity_in_range(self):
        assert _wrap_angle(1.5) == pytest.approx(1.5, abs=1e-6)
