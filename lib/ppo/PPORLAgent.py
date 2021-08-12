import numpy as np
import abc
import torch
import torch.nn as nn
import os

from lib.BaseRLAgent import BaseRLAgent
from lib.policy import BasePolicy


class PPORLAgent(BaseRLAgent):
    """
    Proximal policy approximation agent
    TBD: link paper
    """

    def __init__(
            self,
            policy: BasePolicy,
            discount_rate: float = .99,
            epsilon: float = 0.1,
            epsilon_decay: float = .999,
            beta: float = .01,
            beta_deay: float = .995,
            learning_rate: float = 1e-3,
            SGD_epoch: int = 4,
            device = "cpu",
    ):
        self.beta_deay = beta_deay
        self.epsilon_decay = epsilon_decay
        self.policy = policy
        self.SGD_epoch = SGD_epoch
        self.beta = beta
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        super(PPORLAgent, self).__init__(models=[self.policy], device=device, learning_rate=learning_rate)

        self.policy_optimizer = self._get_optimizer(self.policy.parameters())

    def act(self, states: np.ndarray) -> np.ndarray:
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions, log_probs = self.policy(states)
        return actions.detach().numpy()

    def learn(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            action_probs: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            dones: np.ndarray):

        for _ in range(self.SGD_epoch):
            loss = -self._clipped_surrogate_function(action_probs, states, rewards)
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

        # the clipping parameter reduces as time goes on
        self.epsilon *= self.epsilon_decay

        # the regulation term also reduces
        # this reduces exploration in later runs
        self.beta *= self.beta_deay

    def get_name(self) -> str:
        return "PPO"

    def _clipped_surrogate_function(
            self,
            old_probs: np.ndarray,
            states: np.ndarray,
            rewards: np.ndarray):
        """
        Calculate the clipped surrogate loss function

        :param old_probs: probabilities of original trajectories [trajectories x time_steps]
        :param states: states of original trajectories [trajectories x states]
        :param rewards: rewards of original trajectories [trajectories x rewards]
        :return:
        """

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(self.device)

        actions, log_probs = self.policy(states)
        new_probs = torch.exp(log_probs)

        indices = torch.linspace(rewards.shape[0] - 1, 0, rewards.shape[0]).to(torch.float32).to(self.device)
        reversed_indices = torch.linspace(rewards.shape[0] - 1, 0, rewards.shape[0]).to(torch.long).to(self.device)

        discounts = (self.discount_rate ** indices).view(-1, 1)

        discounted_rewards = discounts * rewards

        future_rewards = discounted_rewards[reversed_indices].cumsum(dim=0)[reversed_indices].to(torch.float)

        mean = future_rewards.mean(dim=1).view(-1, 1)
        std = (future_rewards.std(dim=1) + 1.e-10).view(-1, 1)

        rewards_normalized = (future_rewards - mean) / std

        ratio = new_probs / old_probs
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        clipped_surrogate = torch.min(ratio * rewards_normalized, clipped_ratio * rewards_normalized)

        # include a regularization term
        # this steers new_policy towards 0.5
        # which prevents policy to become exactly 0 or 1
        # this helps with exploration
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) +
                    (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

        return (clipped_surrogate + self.beta * entropy).mean()
