import numpy as np
import torch

from lib.agent.ppo.PPORLAgent import PPORLAgent
from lib.policy.StochasticBasePolicy import StochasticBasePolicy
from lib.function.StateValueFunction import StateValueFunction


class PPO_ActorCriticRLAgent(PPORLAgent):

    def __init__(
            self,
            actor: StochasticBasePolicy,
            critic: StateValueFunction,
            discount_rate: float = .99,
            epsilon: float = 0.1,
            epsilon_decay: float = .999,
            beta: float = .01,
            beta_decay: float = .995,
            learning_rate: float = 1e-3,
            SGD_epoch: int = 4,
            gae_lambda: float = 0.95,
            critic_loss_coefficient: float = 0.5,
            device="cpu",
    ):
        """
        Proximal policy approximation agent, actor-critic style
        https://arxiv.org/abs/1707.06347

        @param actor: The actor network (policy)
        @param critic: The critic network (Value function approximation)
        @param discount_rate: Discounting factor for future rewards (γ)
        @param epsilon: Clipping parameter for the ppo algorithm (ε)
        @param epsilon_decay: Decay factor for epsilon parameter
        @param beta: Coefficient for the entropy term in loss function (β)
                     Encourages the exploration of the action space
        @param beta_decay: Decay factor for the beta parameter
        @param learning_rate: Learning rate for the Adam optimizer (α)
        @param SGD_epoch: Number of repeated optimization steps in the ppo algorithm (K)
        @param gae_lambda: Generalized advantage estimation weighting parameter (λ)
                           λ = 0 recovers temporal difference and λ=1 the monte carlo estimate
        @param device: the device on which the calculations are to be executed
        """
        super(PPO_ActorCriticRLAgent, self).__init__(
            policy=actor,
            discount_rate=discount_rate,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            beta=beta,
            beta_decay=beta_decay,
            learning_rate=learning_rate,
            SGD_epoch=SGD_epoch,
            device=device)

        self.critic_loss_coefficient = critic_loss_coefficient
        self.gae_lambda = gae_lambda
        self.actor = self.policy
        self.critic = critic.to(device)

        self.actor_optimizer = self.policy_optimizer
        self.critic_optimizer = self._get_optimizer(self.critic.parameters())
        self.models = [self.actor, self.critic]
        self.model_names = ["actor", "critic"]

        self.loss = None
        self.critic_loss = None
        self.actor_loss = None

    def act(self, states: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        return super(PPO_ActorCriticRLAgent, self).act(states)

    def learn(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            action_logits: np.ndarray,
            action_log_probs: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        action_logits = torch.tensor(action_logits, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        action_log_probs = torch.tensor(action_log_probs, dtype=torch.float32).to(self.device)

        future_discounted_rewards = self.get_discounted_future_rewards(rewards)

        for _ in range(self.SGD_epoch):
            shape = states.shape
            last_next_state = next_states[:, -1].view(shape[0], 1, -1)

            all_states = torch.cat((next_states, last_next_state), dim=1)

            estimates = self.critic(all_states.reshape(-1, shape[-1])).view(shape[0], shape[1]+1)
            estimated_state_values = estimates[:, :-1]
            estimated_next_state_values = estimates[:, 1:]

            value_last_next_state = estimated_next_state_values[:, -1]
            self.critic_loss = self.critic_loss_coefficient * \
                          (
                                  (
                                          (future_discounted_rewards + value_last_next_state.view(-1, 1))
                                          - estimated_state_values
                                  ) ** 2
                          ).mean()

            advantage = self.estimate_advantages(estimated_state_values=estimated_state_values,
                                                 estimated_next_state_values=estimated_next_state_values,
                                                 rewards=rewards)

            self.actor_loss = -self.clipped_surrogate_function(old_log_probs=action_log_probs, states=states,
                                                          action_logits=action_logits,
                                                          future_discounted_rewards=advantage)

            self.loss = self.actor_loss + self.critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # the clipping parameter reduces as time goes on
        self.epsilon *= self.epsilon_decay

        # the regulation term also reduces
        # this reduces exploration in later runs
        self.beta *= self.beta_deay

    def estimate_advantages(self, estimated_state_values, estimated_next_state_values, rewards):
        """
        Estimate advantages for each (state, next_state, reward) tuple

        @param estimated_state_values: estimated values for state at time step t [trajectories, time steps]
        @param estimated_next_state_values: estimated values for state at time step t+1 [trajectories, time steps]
        @param rewards: received reward at time step t [trajectories, time steps]
        @return: generalized advantage estimation [trajectories, time steps]
        """
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

    def get_log_dict(self) -> dict:
        return {
            "logvar_mean": self.actor.logvar.detach().cpu().numpy().mean(),
            "beta": self.beta,
            "epsilon": self.epsilon,
            "critic_loss": self.critic_loss.detach().cpu().numpy() if self.critic_loss is not None else None,
            "actor_loss": self.actor_loss.detach().cpu().numpy() if self.actor_loss is not None else None,
            "loss": self.loss.detach().cpu().numpy() if self.loss is not None else None,
            "grad_actor":
                np.array([x.grad.norm(dim=0).mean().detach().cpu().numpy() for x in self.actor.parameters()]).mean()
                if self.actor_loss is not None else None,
            "grad_critic":
                np.array([x.grad.norm(dim=0).mean().detach().cpu().numpy() for x in self.critic.parameters()]).mean()
                if self.critic_loss is not None else None
        }
