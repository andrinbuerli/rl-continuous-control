import abc
import torch
import torch.nn as nn
from typing import Callable

from lib.policy.BasePolicy import BasePolicy


class DiscretePolicy(BasePolicy):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 output_transform: Callable[[torch.Tensor], torch.Tensor] = None):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__(state_size=state_size, action_size=action_size, seed=seed, output_transform=output_transform)

        self.policy_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax()
        )

    def get_action_distribution(self, states: torch.Tensor) -> torch.distributions.Distribution:
        probs = self.policy_network(states.to(torch.float32))
        dist = torch.distributions.Categorical(probs)
        return dist