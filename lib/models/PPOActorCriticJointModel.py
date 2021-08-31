from typing import Callable
import torch
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
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        self.v_head = nn.Linear(256, 1)
        self.mu_head = nn.Linear(256, action_size)
        self.variance = nn.Parameter(torch.zeros(action_size))

    def forward(self, states: torch.Tensor):
        x = self.feature_extractor(states.to(torch.float32))
        mu = torch.tanh(self.mu_head(x))
        v = torch.relu(self.v_head(x))
        # we restrict variance to range (0, 1)
        variance = torch.sigmoid(self.variance)
        cov_matrix = torch.diag_embed(variance)
        dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov_matrix)
        actions = dist.sample()
        return {
            "actions": actions,
            "actions_mode": mu,
            "v": v,
            "dist": dist
        }
