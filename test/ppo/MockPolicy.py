import torch
from torch import nn as nn

from lib.policy.BasePolicy import BasePolicy


class MockPolicy(BasePolicy):

    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int,
            return_forward_values):
        super().__init__(state_size=state_size, action_size=action_size, seed=seed)

        self.return_forward_values = return_forward_values
        self.mock_layer = nn.Linear(64, action_size)

    def forward(self, states: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return self.return_forward_values
