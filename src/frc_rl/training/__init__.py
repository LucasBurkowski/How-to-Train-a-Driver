"""Training sub-package."""
from frc_rl.training.trainer import PPOTrainer
from frc_rl.training.self_play import SelfPlayManager

__all__ = ["PPOTrainer", "SelfPlayManager"]
