import numpy as np
import torch

import copy
from typing import Callable

from lib.ppo.PPORLAgent import PPORLAgent
from lib.policy.BasePolicy import BasePolicy
from lib.function.ValueFunction import ValueFunction


class PPO_ActorCriticRLAgent(PPORLAgent):
    """
    Proximal policy approximation agent, actor-critic style
    TBD: link paper
    """

    def __init__(
            self,
            get_actor: Callable[[], BasePolicy],
            get_critic: Callable[[], ValueFunction],
            discount_rate: float = .99,
            epsilon: float = 0.1,
            epsilon_decay: float = .999,
            beta: float = .01,
            beta_deay: float = .995,
            learning_rate: float = 1e-3,
            SGD_epoch: int = 4,
            gae_lambda: float = 0.95,
            tau_soft_update: float = 1e-3,
            device="cpu",
    ):
        super(PPO_ActorCriticRLAgent, self).__init__(
            policy=get_actor(),
            discount_rate=discount_rate,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            beta=beta,
            beta_deay=beta_deay,
            learning_rate=learning_rate,
            SGD_epoch=SGD_epoch,
            device=device)

        self.tau_soft_update = tau_soft_update
        self.gae_lambda = gae_lambda
        self.actor_local = self.policy
        self.critic_local = get_critic().to(device)

        # self.actor_target = get_actor().to(device)
        # self.critic_target = get_critic().to(device)

        self.actor_optimizer = self.policy_optimizer
        self.critic_optimizer = self._get_optimizer(self.critic_local.parameters())
        self.models = [self.actor_local, self.critic_local] #, self.actor_target, self.critic_target]

    def act(self, states: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        return super(PPO_ActorCriticRLAgent, self).act(states)

    def learn(
            self,
            states: np.ndarray,
            action_logits: np.ndarray,
            action_log_probs: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            dones: np.ndarray):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        action_logits = torch.tensor(action_logits, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        action_log_probs = torch.tensor(action_log_probs, dtype=torch.float32).to(self.device)

        future_discounted_rewards = self.get_discounted_future_rewards(rewards)

        for _ in range(self.SGD_epoch):
            shape = states.shape
            estimated_state_values = self.critic_local(states.reshape(-1, shape[-1])).view(shape[0], shape[1])
            estimated_next_state_values = self.critic_local(next_states.reshape(-1, shape[-1])).view(shape[0], shape[1])
            value_last_next_state = estimated_next_state_values[:, -1]
            critic_loss = (((future_discounted_rewards + value_last_next_state.view(-1, 1)) - estimated_state_values) ** 2).mean()

            advantage = self.estimate_advantages(estimated_state_values=estimated_state_values,
                                                 estimated_next_state_values=estimated_next_state_values,
                                                 rewards=rewards)

            actor_loss = -self.clipped_surrogate_function(old_log_probs=action_log_probs, states=states,
                                                          action_logits=action_logits,
                                                          future_discounted_rewards=advantage)

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # the clipping parameter reduces as time goes on
        self.epsilon *= self.epsilon_decay

        # the regulation term also reduces
        # this reduces exploration in later runs
        self.beta *= self.beta_deay

    def estimate_advantages(self, estimated_state_values, estimated_next_state_values, rewards):
        temporal_differences = rewards \
                               + self.discount_rate * estimated_next_state_values \
                               - estimated_state_values

        T = estimated_state_values.shape[1]
        advantage_estimation = torch.empty_like(temporal_differences)
        for t in range(T):
            coefficients = ((self.gae_lambda * self.discount_rate) ** torch.arange(0, T - t, 1)).to(self.device)
            advantage_estimation[:, t] = (temporal_differences[:, t:] * coefficients).sum(dim=1)

        return advantage_estimation

    def get_name(self) -> str:
        return "PPO_ActorCritic"


    #def soft_update(self, local_model, target_model):
    #    """
    #    Soft update model parameters.
    #    θ_target = τ*θ_local + (1 - τ)*θ_target
    #
    #    @param local_model: weights will be copied from
    #    @param target_model: weights will be copied to
    #    @return:
    #    """
    #
    #    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #        target_param.data.copy_(self.tau_soft_update * local_param.data
    #                                + (1.0 - self.tau_soft_update) * target_param.data)
