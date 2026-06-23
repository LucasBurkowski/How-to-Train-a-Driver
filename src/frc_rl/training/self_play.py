"""
Self-play setup for FRC RL.

The FRCEnv returns a joint observation of shape (NUM_ROBOTS * OBS_DIM,)
and expects a joint action of shape (NUM_ROBOTS * ACT_DIM,).

During self-play training both alliances share the same policy.
The SelfPlayWrapper decomposes the joint obs into per-robot slices,
queries the current policy for each robot independently, and
re-assembles the joint action.

This allows StableBaselines3's PPO to train a single policy that
generalises to all six robots on the field simultaneously.

Usage
-----
    env = FRCEnv(...)
    wrapper = SelfPlayWrapper(env)
    # wrapper behaves as a single-agent Gymnasium env whose observation
    # is the PER-ROBOT obs (42-dim) and action is per-robot (3-dim).
    # The wrapper cycles through robots using VecEnv-style batching.

For multi-env training with SB3 SubprocVecEnv, each subprocess runs
a full FRCEnv; the policy sees batches of per-robot observations.

SelfPlayManager
---------------
Provides utilities for:
  - creating a VecEnv of FRCEnv instances
  - periodically snapshotting the current policy as an opponent
  - loading opponent policies for evaluation
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

# Lazy import of SB3 to allow the module to be imported without SB3 installed
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not installed; SelfPlayManager disabled.")


# Number of robots and per-robot dimensions (must match frc_env.py)
NUM_ROBOTS = 6
OBS_DIM = 27
ACT_DIM = 3


# ---------------------------------------------------------------------------
# SelfPlayWrapper
# ---------------------------------------------------------------------------

class SelfPlayWrapper(gym.Wrapper):
    """
    Converts the 6-robot joint env into a single-agent per-robot env.

    The wrapper presents a single robot's observation / action at a time.
    All NUM_ROBOTS robots are stepped together; the total reward is the
    average per-robot reward (so the policy learns behaviour beneficial
    for each individual robot).

    Observation space : Box(OBS_DIM,)  – one robot's observation
    Action space      : Box(ACT_DIM,)  – one robot's velocity commands
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Override obs / action spaces to per-robot sizes
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        self._joint_obs: Optional[np.ndarray] = None
        self._robot_cursor: int = 0          # which robot we're currently serving
        self._pending_actions: np.ndarray = np.zeros(NUM_ROBOTS * ACT_DIM, dtype=np.float32)
        self._accumulated_reward: float = 0.0
        self._last_info: Dict[str, Any] = {}
        self._terminated = False
        self._truncated = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        joint_obs, info = self.env.reset(seed=seed, options=options)
        self._joint_obs = joint_obs
        self._robot_cursor = 0
        self._pending_actions[:] = 0.0
        self._accumulated_reward = 0.0
        self._terminated = False
        self._truncated = False
        return self._current_robot_obs(), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Store this robot's action
        start = self._robot_cursor * ACT_DIM
        self._pending_actions[start : start + ACT_DIM] = action

        self._robot_cursor += 1

        if self._robot_cursor < NUM_ROBOTS:
            # Not all robots have acted yet; return current obs without stepping
            obs = self._current_robot_obs()
            return obs, 0.0, False, False, {}

        # All robots have acted → step the underlying env
        joint_obs, reward, terminated, truncated, info = self.env.step(
            self._pending_actions.copy()
        )
        self._joint_obs = joint_obs
        self._robot_cursor = 0
        self._terminated = terminated
        self._truncated = truncated
        self._last_info = info

        # Return per-robot avg reward
        per_robot_rewards: List[float] = info.get("robot_rewards", [reward / NUM_ROBOTS] * NUM_ROBOTS)
        avg_reward = float(np.mean(per_robot_rewards))

        return self._current_robot_obs(), avg_reward, terminated, truncated, info

    def _current_robot_obs(self) -> np.ndarray:
        if self._joint_obs is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        idx = self._robot_cursor % NUM_ROBOTS
        return self._joint_obs[idx * OBS_DIM : (idx + 1) * OBS_DIM].copy()


# ---------------------------------------------------------------------------
# SelfPlayManager
# ---------------------------------------------------------------------------

class SelfPlayManager:
    """
    Manages self-play training: environment creation, policy snapshots, etc.

    Parameters
    ----------
    env_factory : Callable[[], gym.Env]
        Zero-argument factory that returns a fresh FRCEnv instance.
    snapshot_dir : str
        Directory where policy snapshots are stored.
    snapshot_interval_steps : int
        Save a new snapshot every N training steps.
    """

    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        snapshot_dir: str = "snapshots",
        snapshot_interval_steps: int = 100_000,
    ) -> None:
        self.env_factory = env_factory
        self.snapshot_dir = snapshot_dir
        self.snapshot_interval_steps = snapshot_interval_steps
        os.makedirs(snapshot_dir, exist_ok=True)
        self._snapshots: List[str] = []

    def make_env(self) -> SelfPlayWrapper:
        """Return a SelfPlayWrapper around a fresh FRCEnv."""
        return SelfPlayWrapper(self.env_factory())

    def make_vec_env(self, n_envs: int = 4) -> "VecEnv":
        """Return a SubprocVecEnv of SelfPlayWrapper environments."""
        if not _SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 is required for make_vec_env.")
        return make_vec_env(
            self.make_env,
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
        )

    def save_snapshot(self, policy: "PPO", step: int) -> str:
        """Save a policy snapshot; return the snapshot path."""
        path = os.path.join(self.snapshot_dir, f"policy_step_{step}.zip")
        policy.save(path)
        self._snapshots.append(path)
        logger.info("Saved policy snapshot: %s", path)
        return path

    def load_latest_snapshot(self) -> Optional[str]:
        """Return path to the most recently saved snapshot, or None."""
        if not self._snapshots:
            # Check disk
            import glob
            files = sorted(glob.glob(os.path.join(self.snapshot_dir, "policy_step_*.zip")))
            if files:
                self._snapshots = files
        return self._snapshots[-1] if self._snapshots else None

    @property
    def snapshot_paths(self) -> List[str]:
        return list(self._snapshots)
