#!/usr/bin/env python3
"""
Training entry point for the FRC RL Driver.

Usage
-----
# Train with defaults:
    python scripts/train.py

# Train with custom config:
    python scripts/train.py --config configs/default_config.yaml

# Resume from checkpoint:
    python scripts/train.py --checkpoint snapshots/frc_ppo_final.zip

# Connect to live WPILib maple-sim simulation:
    python scripts/train.py --bridge-ip localhost
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frc_rl.training.trainer import PPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the FRC PPO driving model."
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: use built-in defaults).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .zip checkpoint to resume training from.",
    )
    p.add_argument(
        "--bridge-ip",
        type=str,
        default=None,
        help="IP of WPILib maple-sim server.  Omit for standalone Python sim.",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total training timesteps.",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments.",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs.",
    )
    p.add_argument(
        "--snapshot-dir",
        type=str,
        default=None,
        help="Directory for policy snapshots.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {}
    if args.checkpoint:
        overrides["checkpoint_path"] = args.checkpoint
    if args.bridge_ip:
        overrides["wpilib_bridge_ip"] = args.bridge_ip
    if args.timesteps:
        overrides["total_timesteps"] = args.timesteps
    if args.n_envs:
        overrides["n_envs"] = args.n_envs
    if args.log_dir:
        overrides["log_dir"] = args.log_dir
    if args.snapshot_dir:
        overrides["snapshot_dir"] = args.snapshot_dir

    trainer = PPOTrainer(config_path=args.config, **overrides)
    logger.info("Starting training with config: %s", trainer.config)
    model = trainer.train()
    logger.info("Training finished.  Model: %s", model)


if __name__ == "__main__":
    main()
