"""Tests for the FRC Gymnasium environment."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frc_rl.env.frc_env import FRCEnv, NUM_ROBOTS, OBS_DIM, ACT_DIM
from frc_rl.env.field import FRCField


@pytest.fixture
def env() -> FRCEnv:
    e = FRCEnv(field=FRCField.reefscape_2025())
    yield e
    e.close()


class TestEnvSpaces:
    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (NUM_ROBOTS * OBS_DIM,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (NUM_ROBOTS * ACT_DIM,)

    def test_action_space_bounds(self, env):
        assert (env.action_space.low == -1.0).all()
        assert (env.action_space.high == 1.0).all()


class TestEnvReset:
    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (NUM_ROBOTS * OBS_DIM,)
        assert isinstance(info, dict)

    def test_reset_obs_within_bounds(self, env):
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs))

    def test_reset_step_count_zero(self, env):
        env.reset()
        assert env._step_count == 0

    def test_reset_game_pieces_active(self, env):
        env.reset()
        active = [p for p in env._game_pieces if p.active]
        assert len(active) > 0


class TestEnvStep:
    def test_step_zero_action_returns_finite_obs(self, env):
        env.reset()
        action = np.zeros(NUM_ROBOTS * ACT_DIM, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert math.isfinite(reward)

    def test_step_increments_step_count(self, env):
        env.reset()
        env.step(np.zeros(NUM_ROBOTS * ACT_DIM, dtype=np.float32))
        assert env._step_count == 1

    def test_step_info_contains_robot_rewards(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(NUM_ROBOTS * ACT_DIM, dtype=np.float32))
        assert "robot_rewards" in info
        assert len(info["robot_rewards"]) == NUM_ROBOTS

    def test_step_info_contains_alliance_scores(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(NUM_ROBOTS * ACT_DIM, dtype=np.float32))
        assert "alliance_scores" in info
        assert "red" in info["alliance_scores"]
        assert "blue" in info["alliance_scores"]

    def test_episode_terminates(self, env):
        env.reset()
        # Run until termination
        terminated = truncated = False
        for _ in range(env.max_steps + 10):
            _, _, terminated, truncated, _ = env.step(
                np.zeros(NUM_ROBOTS * ACT_DIM, dtype=np.float32)
            )
            if terminated or truncated:
                break
        assert terminated or truncated

    def test_action_clipping(self, env):
        """Out-of-range actions should not raise errors."""
        env.reset()
        big_action = np.full(NUM_ROBOTS * ACT_DIM, 100.0, dtype=np.float32)
        obs, _, _, _, _ = env.step(big_action)
        assert np.all(np.isfinite(obs))

    def test_robots_stay_within_field(self, env):
        from frc_rl.env.field import FIELD_LENGTH, FIELD_WIDTH

        env.reset()
        # Apply max forward velocity for multiple steps
        action = np.zeros(NUM_ROBOTS * ACT_DIM, dtype=np.float32)
        action[0::ACT_DIM] = 1.0  # all robots max +vx

        for _ in range(100):
            env.step(action)

        for state in env.robot_states:
            assert 0.0 <= state.x <= FIELD_LENGTH, f"x={state.x} out of bounds"
            assert 0.0 <= state.y <= FIELD_WIDTH, f"y={state.y} out of bounds"

    def test_seeded_reset_deterministic(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestEnvMultipleEpisodes:
    def test_multiple_resets(self, env):
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            assert np.all(np.isfinite(obs))
