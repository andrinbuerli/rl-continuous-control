import abc
from typing import Callable
import torch
import torch.nn as nn

from lib.policy.BasePolicy import BasePolicy


class ContinuousDiagonalGaussianPolicy(BasePolicy):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int,
            output_transform: Callable[[torch.Tensor], torch.Tensor] = None):
        """
        Policy which learns to sample an action from a continuous multivariate gaussian distribution where each
        action dimension is considered to be independent.

        @param state_size: Dimension of each state
        @param action_size: Dimension of each action
        @param seed: Random seed
        @param output_transform: optional, generic transformation to be applied to policy output
        """
        super().__init__(state_size=state_size, action_size=action_size, seed=seed, output_transform=output_transform)
        self.output_transform = output_transform

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)

        self.mu_head = nn.Linear(64, action_size)
        self.diagonal_sigma_head = nn.Linear(64, action_size)

    def get_action_distribution(self, states: torch.Tensor) -> torch.distributions.Distribution:
        x1 = torch.relu(self.fc1(states.to(torch.float32)))
        x = torch.relu(self.fc2(x1))
        mu = self.mu_head(x)
        # sigma must not be smaller than 0, so we interpret the output as ln(sigma)
        diag_sigma = torch.exp(self.diagonal_sigma_head(x))
        dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=torch.diag_embed(diag_sigma))
        return dist
