import abc
import torch
import torch.nn as nn


class BasePolicy(nn.Module):

    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)

    @abc.abstractmethod
    def forward(self, states: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
         Determine next action based on the current states

        @param states: The current states
        @return: (next actions, log probabilities of actions)
        """
        pass
