#!/usr/bin/env python3
"""
Evaluation entry point for a trained FRC RL Driver model.

Usage
-----
    python scripts/evaluate.py --model snapshots/frc_ppo_final.zip
    python scripts/evaluate.py --model snapshots/frc_ppo_final.zip --episodes 20 --render
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frc_rl.training.trainer import PPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained FRC PPO driving model."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to a saved .zip model.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10).",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Print ASCII match state each step.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config (for env settings).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trainer = PPOTrainer(config_path=args.config)
    stats = trainer.evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
    )
    logger.info("Evaluation results over %d episodes:", args.episodes)
    for k, v in stats.items():
        logger.info("  %-25s : %.4f", k, v)


if __name__ == "__main__":
    main()
