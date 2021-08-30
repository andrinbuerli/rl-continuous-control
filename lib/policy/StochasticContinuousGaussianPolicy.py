from typing import Callable
import torch
import torch.nn as nn
from torch.functional import F

from lib.policy.StochasticBasePolicy import StochasticBasePolicy


class StochasticContinuousGaussianPolicy(StochasticBasePolicy):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int,
            output_transform: Callable[[torch.Tensor], torch.Tensor] = None,
            reduced_capacity: bool = True):
        """
        Stochastic policy which learns to sample an action from a continuous multivariate gaussian distribution where
        each action dimension is considered to be independent.
        """
        super().__init__(state_size=state_size, action_size=action_size, seed=seed, output_transform=output_transform)
        self.output_transform = output_transform

        if reduced_capacity:
            self.policy_network = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
        else:
            self.policy_network = nn.Sequential(
                nn.Linear(state_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )

        self.mu_head = nn.Linear(64, action_size)
        self.variance = nn.Parameter(torch.ones(action_size)*0.15)

    def get_action_distribution(self, states: torch.Tensor) -> torch.distributions.Distribution:
        x = self.policy_network(states.to(torch.float32))
        mu = F.tanh(self.mu_head(x))
        # we restrict sigma to range -1, 1 as well
        variance = F.tanh(self.variance)
        cov_matrix = torch.diag_embed(variance)
        dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov_matrix)
        return dist
