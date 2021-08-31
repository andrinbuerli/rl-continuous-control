import abc
import torch
import torch.nn as nn


class StateValueFunction(nn.Module):

    def __init__(
            self,
            state_size: int,
            seed: int,
            reduced_capacity: bool = True):
        super().__init__()
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)

        if reduced_capacity:
            self.value_function_network = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:
            self.value_function_network = nn.Sequential(
                nn.Linear(state_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_function_network(states)