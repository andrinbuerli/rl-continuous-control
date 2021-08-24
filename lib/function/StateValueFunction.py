import abc
import torch
import torch.nn as nn


class StateValueFunction(nn.Module):

    def __init__(
            self,
            state_size: int,
            seed: int):
        super().__init__()
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)

        self.value_function_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    @abc.abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_function_network(states)
