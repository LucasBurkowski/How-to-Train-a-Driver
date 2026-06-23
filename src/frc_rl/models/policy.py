"""
Custom policy network for the FRC RL agent.

Architecture
------------
Input: flat observation vector (OBS_DIM = 27 per robot, but the
       StableBaselines3 PPO policy sees the *per-robot* slice).

Two-tower design:
  - ego_tower     : MLP processing own-robot features (indices 0..6)
  - context_tower : MLP processing game state features (indices 7..)
  -> concat -> shared_head -> policy / value outputs

This is registered with SB3 as a custom ActorCriticPolicy so PPO
can train it end-to-end.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class FRCFeaturesExtractor(BaseFeaturesExtractor):
    """
    Two-tower MLP feature extractor for FRC observations.

    The flat observation is split into:
      ego_features    : first 7 dims  (own pose + velocity + has_piece)
      context_features: remaining dims (piece/zone directions + relative poses)
    """

    EGO_DIM = 7
    FEATURES_DIM = 128

    def __init__(self, observation_space: gym.spaces.Box) -> None:
        super().__init__(observation_space, features_dim=self.FEATURES_DIM)

        obs_dim = int(observation_space.shape[0])
        context_dim = obs_dim - self.EGO_DIM

        # Ego tower
        self.ego_tower = nn.Sequential(
            nn.Linear(self.EGO_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Context tower
        self.context_tower = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Shared head
        self.shared_head = nn.Sequential(
            nn.Linear(128, self.FEATURES_DIM),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ego = observations[:, : self.EGO_DIM]
        context = observations[:, self.EGO_DIM :]
        ego_feat = self.ego_tower(ego)
        ctx_feat = self.context_tower(context)
        combined = torch.cat([ego_feat, ctx_feat], dim=1)
        return self.shared_head(combined)


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------

class FRCPolicyNetwork(ActorCriticPolicy):
    """
    Custom ActorCriticPolicy using the two-tower FRCFeaturesExtractor.

    This is passed directly to `stable_baselines3.PPO(policy=FRCPolicyNetwork)`.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch: Optional[List[Dict]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch or [{"pi": [64, 64], "vf": [64, 64]}],
            features_extractor_class=FRCFeaturesExtractor,
            features_extractor_kwargs={},
            **kwargs,
        )
