import abc
import torch
import torch.nn as nn

from lib.policy.BasePolicy import BasePolicy


class DiscretePolicy(BasePolicy):

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__(state_size=state_size, action_size=action_size, seed=seed)

        self.policy_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax()
        )

    def forward(self, states) -> (torch.Tensor, torch.Tensor):
        probs = self.policy_network(states.to(torch.float32))
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        return actions, dist.log_prob(actions)
