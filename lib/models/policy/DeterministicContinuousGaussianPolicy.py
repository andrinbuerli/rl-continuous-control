from typing import Callable
import torch
import torch.nn as nn

from lib.models.policy import DeterministicBasePolicy


class DeterministicContinuousGaussianPolicy(DeterministicBasePolicy):
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
        super().__init__(state_size=state_size, action_size=action_size, seed=seed)

        self.output_transform = output_transform
        if reduced_capacity:
            self.policy_network = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_size)
            )
        else:
            self.policy_network = nn.Sequential(
                nn.Linear(state_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, self.action_size)
            )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.policy_network(states.to(torch.float32))
        if self.output_transform is not None:
            return self.output_transform(x)
        else:
            return x
