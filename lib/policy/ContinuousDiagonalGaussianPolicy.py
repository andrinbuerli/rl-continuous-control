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
        super().__init__(state_size=state_size, action_size=action_size, seed=seed)
        self.output_transform = output_transform

        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.mu_head = nn.Linear(64, action_size)
        self.diagonal_sigma_head = nn.Linear(64, action_size)

    def forward(self, states) -> (torch.Tensor, torch.Tensor):
        features = self.feature_extractor(states.to(torch.float32))
        mu = self.mu_head(features)
        # sigma must not be smaller than 0, so we interpret the output as ln(sigma)
        diag_sigma = torch.exp(self.diagonal_sigma_head(features))

        dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=torch.diag_embed(diag_sigma))
        raw_actions = dist.sample()
        if self.output_transform is not None:
            actions = self.output_transform(raw_actions)
        else:
            actions = raw_actions

        return actions, dist.log_prob(raw_actions)
