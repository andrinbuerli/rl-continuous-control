import numpy as np
import torch

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
        self.policy = policy.to(device)
        self.SGD_epoch = SGD_epoch
        self.beta = beta
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        super(PPORLAgent, self).__init__(models=[self.policy], device=device, learning_rate=learning_rate)

        self.policy_optimizer = self._get_optimizer(self.policy.parameters())

    def act(self, states: np.ndarray) -> (np.ndarray, np.ndarray):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions, action_logits, dist = self.policy(states)
        return actions.detach().cpu().numpy(), action_logits.detach().cpu().numpy(), dist.log_prob(action_logits).detach().cpu().numpy()

    def learn(
            self,
            states: np.ndarray,
            action_logits: np.ndarray,
            action_log_probs: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            dones: np.ndarray):

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        action_logits = torch.tensor(action_logits, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        action_log_probs = torch.tensor(action_log_probs, dtype=torch.float32).to(self.device)

        rewards = self.get_discounted_future_rewards(rewards)

        for _ in range(self.SGD_epoch):
            loss = -self.clipped_surrogate_function(old_log_probs=action_log_probs, states=states,
                                                    action_logits=action_logits, future_discounted_rewards=rewards)
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

    def clipped_surrogate_function(
            self,
            action_logits: np.ndarray,
            old_log_probs: np.ndarray,
            states: np.ndarray,
            future_discounted_rewards: np.ndarray):
        """
        Calculate the clipped surrogate loss function

        :param old_log_probs: probabilities of original trajectories [trajectories x time_steps]
        :param states: states of original trajectories [trajectories x states]
        :param future_discounted_rewards: rewards of original trajectories [trajectories x rewards]
        :return: differentiable loss
        """

        shape = states.shape
        dist = self.policy.get_action_distribution(states.reshape(-1, shape[-1]))
        new_log_probs, entropy = dist.log_prob(action_logits.reshape(-1, action_logits.shape[-1])), dist.entropy()
        new_log_probs, entropy = new_log_probs.view(shape[0], shape[1]), entropy.view(shape[0], shape[1])

        if future_discounted_rewards.shape[0] > 1:
            mean = future_discounted_rewards.mean(dim=0).view(1, -1)
            std = (future_discounted_rewards.std(dim=0) + 1.e-10).view(1, -1)

            rewards_normalized = (future_discounted_rewards - mean) / std
        else:
            rewards_normalized = future_discounted_rewards

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        clipped_surrogate = torch.min(ratio * rewards_normalized, clipped_ratio * rewards_normalized)

        # include a regularization term
        # this steers new_policy towards a high entropy state
        # this helps with exploration
        regularization = self.beta * entropy

        return (clipped_surrogate + regularization).mean()

    def get_discounted_future_rewards(self, rewards):
        indices = torch.linspace(0, rewards.shape[1] - 1, rewards.shape[1]).to(torch.float32).to(self.device)
        reversed_indices = torch.linspace(rewards.shape[1] - 1, 0, rewards.shape[1]).to(torch.long).to(self.device)
        discounts = (self.discount_rate ** indices).view(1, -1)
        discounted_rewards = discounts * rewards
        future_rewards = discounted_rewards[:, reversed_indices].cumsum(dim=1)[:, reversed_indices].to(torch.float)
        return future_rewards
