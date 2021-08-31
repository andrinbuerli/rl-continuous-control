from typing import Callable
import torch
from torch.functional import F
import torch.nn as nn

from lib.models.policy.StochasticBasePolicy import StochasticBasePolicy


class PPOActorCriticJointModel(nn.Module):

    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)

        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        self.v_head = nn.Linear(256, 1)
        self.mu_head = nn.Linear(256, action_size)
        self.std = nn.Parameter(torch.ones(1, action_size)*0.15)

    def forward(self, states: torch.Tensor, scale: torch.float32 = 1.0):
        x = self.feature_extractor(states.to(torch.float32))
        mu = torch.tanh(self.mu_head(x))
        v = self.v_head(x)
        # we restrict variance to range (0, 1)
        std = F.hardtanh(self.std, min_val=0.06 * scale, max_val=0.6 * scale)
        dist = torch.distributions.Normal(mu, std)
        actions = dist.sample()
        return {
            "actions": actions,
            "actions_mode": mu,
            "v": v,
            "dist": dist
        }
