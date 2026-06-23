"""
PPO Trainer for FRC robot driving models.

Wraps StableBaselines3 PPO with:
  - Custom FRCPolicyNetwork
  - Self-play environment management
  - Periodic policy snapshots
  - TensorBoard / CSV logging
  - Checkpoint resume support
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecEnv

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    logger.error("stable-baselines3 not installed.  PPOTrainer cannot function.")

from frc_rl.env.frc_env import FRCEnv
from frc_rl.env.field import FRCField
from frc_rl.env.reward import RewardConfig
from frc_rl.models.policy import FRCPolicyNetwork
from frc_rl.training.self_play import SelfPlayManager, SelfPlayWrapper


# ---------------------------------------------------------------------------
# Snapshot callback
# ---------------------------------------------------------------------------

class SnapshotCallback(BaseCallback):  # type: ignore[misc]
    """Saves a policy snapshot every N steps via SelfPlayManager."""

    def __init__(
        self,
        manager: SelfPlayManager,
        save_freq: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.manager = manager
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.manager.save_snapshot(self.model, self.n_calls)
        return True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    High-level trainer for the FRC PPO model.

    Parameters
    ----------
    config_path : str | None
        Path to a YAML config file.  Overrides any kwarg defaults.
    **kwargs :
        Config overrides (see DEFAULT_CONFIG).
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        # Training
        "total_timesteps": 10_000_000,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        # Environment
        "dt": 0.02,
        "wpilib_bridge_ip": None,        # None = standalone Python simulation
        # Paths
        "log_dir": "logs/",
        "snapshot_dir": "snapshots/",
        "checkpoint_path": None,         # Resume from checkpoint
        # Self-play
        "snapshot_interval_steps": 100_000,
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 is required for PPOTrainer.")

        self.config = dict(self.DEFAULT_CONFIG)
        if config_path is not None:
            self._load_yaml(config_path)
        self.config.update(kwargs)

        os.makedirs(self.config["log_dir"], exist_ok=True)
        os.makedirs(self.config["snapshot_dir"], exist_ok=True)

        self._manager = SelfPlayManager(
            env_factory=self._make_base_env,
            snapshot_dir=self.config["snapshot_dir"],
            snapshot_interval_steps=self.config["snapshot_interval_steps"],
        )
        self._model: Optional[PPO] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> PPO:
        """Run (or resume) PPO training and return the final model."""
        vec_env = self._manager.make_vec_env(n_envs=self.config["n_envs"])

        if self.config.get("checkpoint_path") and os.path.exists(
            self.config["checkpoint_path"]
        ):
            logger.info("Resuming from checkpoint: %s", self.config["checkpoint_path"])
            model = PPO.load(
                self.config["checkpoint_path"],
                env=vec_env,
                device="auto",
            )
        else:
            model = PPO(
                policy=FRCPolicyNetwork,
                env=vec_env,
                learning_rate=self.config["learning_rate"],
                n_steps=self.config["n_steps"],
                batch_size=self.config["batch_size"],
                n_epochs=self.config["n_epochs"],
                gamma=self.config["gamma"],
                gae_lambda=self.config["gae_lambda"],
                clip_range=self.config["clip_range"],
                ent_coef=self.config["ent_coef"],
                vf_coef=self.config["vf_coef"],
                max_grad_norm=self.config["max_grad_norm"],
                tensorboard_log=self.config["log_dir"],
                verbose=1,
                device="auto",
            )

        self._model = model

        callbacks = [
            CheckpointCallback(
                save_freq=self.config["snapshot_interval_steps"],
                save_path=self.config["snapshot_dir"],
                name_prefix="frc_ppo",
            ),
            SnapshotCallback(
                manager=self._manager,
                save_freq=self.config["snapshot_interval_steps"],
            ),
        ]

        model.learn(
            total_timesteps=self.config["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=self.config.get("checkpoint_path") is None,
            tb_log_name="frc_ppo",
        )

        final_path = os.path.join(self.config["snapshot_dir"], "frc_ppo_final")
        model.save(final_path)
        logger.info("Training complete.  Final model saved to %s", final_path)
        return model

    def evaluate(
        self,
        model_path: str,
        n_episodes: int = 10,
        render: bool = False,
    ) -> Dict[str, float]:
        """Evaluate a saved model and return aggregate statistics."""
        env = SelfPlayWrapper(
            self._make_base_env(render_mode="human" if render else None)
        )
        model = PPO.load(model_path, env=env, device="auto")

        episode_rewards = []
        episode_scores: Dict[str, list] = {"red": [], "blue": []}

        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            episode_rewards.append(total_reward)
            scores = info.get("alliance_scores", {})
            episode_scores["red"].append(scores.get("red", 0.0))
            episode_scores["blue"].append(scores.get("blue", 0.0))

        import numpy as np

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_red_score": float(np.mean(episode_scores["red"])),
            "mean_blue_score": float(np.mean(episode_scores["blue"])),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_base_env(self, render_mode: Optional[str] = None) -> FRCEnv:
        bridge = None
        bridge_ip = self.config.get("wpilib_bridge_ip")
        if bridge_ip:
            from frc_rl.sim.wpilib_bridge import WPILibBridge

            bridge = WPILibBridge(server_ip=bridge_ip)
            if not bridge.connect():
                logger.warning("Could not connect to WPILib bridge; using standalone sim.")
                bridge = None

        return FRCEnv(
            field=FRCField.reefscape_2025(),
            reward_config=RewardConfig(),
            dt=self.config["dt"],
            bridge=bridge,
            render_mode=render_mode,
        )

    def _load_yaml(self, path: str) -> None:
        with open(path) as f:
            data = yaml.safe_load(f)
        if data:
            self.config.update(data)
