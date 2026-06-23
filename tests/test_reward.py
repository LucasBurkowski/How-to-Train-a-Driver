"""Tests for the reward calculator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frc_rl.env.reward import RewardCalculator, RewardConfig, RobotRewardState


@pytest.fixture
def calc() -> RewardCalculator:
    return RewardCalculator(RewardConfig())


@pytest.fixture
def rs() -> RobotRewardState:
    return RobotRewardState(robot_id=0, alliance="blue")


class TestRewardCalculator:
    def test_time_penalty_applied(self, calc, rs):
        reward = calc.compute(rs, {})
        assert reward < 0  # only time penalty

    def test_intake_reward(self, calc, rs):
        reward = calc.compute(rs, {"intake_success": True})
        assert reward > 0
        assert rs.has_game_piece is True

    def test_score_reward(self, calc, rs):
        rs.has_game_piece = True
        reward = calc.compute(rs, {"score_success": True, "score_point_value": 7})
        assert reward > 0
        assert rs.has_game_piece is False
        assert rs.total_scored == 1

    def test_failed_score_penalty(self, calc, rs):
        reward = calc.compute(rs, {"failed_score": True})
        assert reward < -calc.config.failed_score_penalty  # penalty + time

    def test_robot_contact_penalty(self, calc, rs):
        reward = calc.compute(rs, {"robot_contact": True})
        assert reward < -calc.config.robot_contact_penalty

    def test_wall_contact_penalty(self, calc, rs):
        reward = calc.compute(rs, {"wall_contact": True})
        assert reward < -calc.config.wall_contact_penalty

    def test_climb_reward(self, calc, rs):
        reward = calc.compute(rs, {"climb_success": True})
        expected = calc.config.endgame_climb_reward - calc.config.time_penalty
        assert reward == pytest.approx(expected)

    def test_no_double_penalty_intake_and_contact(self, calc, rs):
        # Both events in same step
        reward = calc.compute(
            rs, {"intake_success": True, "robot_contact": True}
        )
        # Net should be: +2 (intake) - 1 (contact) - 0.01 (time) = ~0.99
        assert reward == pytest.approx(
            calc.config.intake_reward
            - calc.config.robot_contact_penalty
            - calc.config.time_penalty,
            abs=1e-6,
        )

    def test_consecutive_failed_scores_tracked(self, calc, rs):
        calc.compute(rs, {"failed_score": True})
        calc.compute(rs, {"failed_score": True})
        assert rs.consecutive_failed_scores == 2

    def test_consecutive_resets_on_score(self, calc, rs):
        rs.consecutive_failed_scores = 3
        rs.has_game_piece = True
        calc.compute(rs, {"score_success": True, "score_point_value": 2})
        assert rs.consecutive_failed_scores == 0
