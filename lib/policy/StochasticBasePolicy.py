import abc
import torch
import torch.nn as nn
from typing import Callable


class StochasticBasePolicy(nn.Module):

    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int,
            output_transform: Callable[[torch.Tensor], torch.Tensor] = None):
        """
        Initialize the policy

        @param state_size: Dimension of each state
        @param action_size: Dimension of each action
        @param seed: Random seed
        @param output_transform: optional, generic transformation to be applied to policy output
        """
        super().__init__()
        self.output_transform = output_transform
        self.action_size = action_size
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)

    def forward(self, states: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.distributions.Distribution):
        """
         Determine next action based on the current states

        @param states: The current states
        @return: (next actions, action logits, policy distribution)
        """
        dist = self.get_action_distribution(states)
        action_logits = dist.sample()

        actions = torch.clip(action_logits, -1, 1)

        # if self.output_transform is not None:
        #     actions = self.output_transform(action_logits)
        # else:
        #     actions = action_logits

        return actions, action_logits, dist

    @abc.abstractmethod
    def get_action_distribution(self, states: torch.Tensor) -> torch.distributions.Distribution:
        """
        Determine the probability distribution over the action space given the state

        @param states: The current states
        @return: The corresponding action distributions
        """
        pass
