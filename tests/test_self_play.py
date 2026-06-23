"""Tests for the self-play wrapper and manager."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frc_rl.env.frc_env import FRCEnv, NUM_ROBOTS, OBS_DIM, ACT_DIM
from frc_rl.env.field import FRCField
from frc_rl.training.self_play import SelfPlayWrapper, SelfPlayManager


@pytest.fixture
def base_env() -> FRCEnv:
    e = FRCEnv(field=FRCField.reefscape_2025())
    yield e
    e.close()


@pytest.fixture
def wrapper(base_env) -> SelfPlayWrapper:
    return SelfPlayWrapper(base_env)


class TestSelfPlayWrapper:
    def test_per_robot_obs_shape(self, wrapper):
        assert wrapper.observation_space.shape == (OBS_DIM,)

    def test_per_robot_action_shape(self, wrapper):
        assert wrapper.action_space.shape == (ACT_DIM,)

    def test_reset_returns_single_robot_obs(self, wrapper):
        obs, _ = wrapper.reset()
        assert obs.shape == (OBS_DIM,)

    def test_full_cycle_all_robots(self, wrapper):
        """Step all NUM_ROBOTS times to complete one joint env step."""
        wrapper.reset()
        action = np.zeros(ACT_DIM, dtype=np.float32)
        rewards = []
        for i in range(NUM_ROBOTS):
            obs, reward, terminated, truncated, info = wrapper.step(action)
            rewards.append(reward)
            assert obs.shape == (OBS_DIM,)
        # Only the last step (completing a joint step) should have non-zero reward
        # (all time penalties accumulate at joint step)
        final_reward = rewards[-1]
        assert isinstance(final_reward, float)

    def test_robot_cursor_wraps(self, wrapper):
        wrapper.reset()
        action = np.zeros(ACT_DIM, dtype=np.float32)
        for _ in range(NUM_ROBOTS):
            wrapper.step(action)
        # After a full cycle, cursor should be reset to 0
        assert wrapper._robot_cursor == 0

    def test_episode_terminates(self, wrapper):
        wrapper.reset()
        action = np.zeros(ACT_DIM, dtype=np.float32)
        terminated = truncated = False
        # Step through max_steps * NUM_ROBOTS individual robot steps
        max_iter = wrapper.env.max_steps * NUM_ROBOTS + NUM_ROBOTS * 2
        for _ in range(max_iter):
            _, _, terminated, truncated, _ = wrapper.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated


class TestSelfPlayManager:
    def test_make_env_returns_wrapper(self):
        manager = SelfPlayManager(
            env_factory=lambda: FRCEnv(field=FRCField.reefscape_2025()),
            snapshot_dir="/tmp/test_snapshots",
        )
        env = manager.make_env()
        assert isinstance(env, SelfPlayWrapper)
        env.close()

    def test_save_snapshot(self, tmp_path):
        manager = SelfPlayManager(
            env_factory=lambda: FRCEnv(field=FRCField.reefscape_2025()),
            snapshot_dir=str(tmp_path),
        )
        mock_model = MagicMock()
        mock_model.save = MagicMock()
        manager.save_snapshot(mock_model, step=1000)
        mock_model.save.assert_called_once()
        assert len(manager.snapshot_paths) == 1

    def test_load_latest_snapshot_empty(self, tmp_path):
        manager = SelfPlayManager(
            env_factory=lambda: FRCEnv(),
            snapshot_dir=str(tmp_path),
        )
        assert manager.load_latest_snapshot() is None
