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
        Stochastic policy which learns to sample an action from a continuous multivariate gaussian distribution where
        each action dimension is considered to be independent.
        """
        super().__init__(state_size=state_size, action_size=action_size, seed=seed, output_transform=output_transform)
        self.output_transform = output_transform

        self.policy_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(64, action_size)
        self.diagonal_sigma_head = nn.Linear(64, action_size)

    def get_action_distribution(self, states: torch.Tensor) -> torch.distributions.Distribution:
        x = self.policy_network(states.to(torch.float32))
        mu = self.mu_head(x)
        # sigma must not be smaller than 0, so we interpret the output as ln(sigma)
        diag_sigma = torch.exp(self.diagonal_sigma_head(x))
        dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=torch.diag_embed(diag_sigma))
        return dist
