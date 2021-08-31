import numpy as np
import torch

from lib.agent.BaseRLAgent import BaseRLAgent
from lib.models.policy.StochasticBasePolicy import StochasticBasePolicy
from lib.models.function import StateValueFunction
from lib.models.PPOActorCriticJointModel import PPOActorCriticJointModel


class PPOActorCriticRLAgent(BaseRLAgent):

    def __init__(
            self,
            model: PPOActorCriticJointModel,
            discount_rate: float = .99,
            epsilon: float = 0.1,
            epsilon_decay: float = .999,
            beta: float = .01,
            beta_decay: float = .995,
            learning_rate: float = 1e-3,
            SGD_epoch: int = 4,
            batch_size: int = 32,
            gae_lambda: float = 0.95,
            critic_loss_coefficient: float = 0.5,
            grad_clip_max: float = None,
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

        self.beta_deay = beta_decay
        self.epsilon_decay = epsilon_decay
        self.model = model.to(device)
        self.SGD_epoch = SGD_epoch
        self.beta = beta
        self.epsilon = epsilon
        self.discount_rate = discount_rate

        super(PPOActorCriticRLAgent, self).__init__(
            models=[self.model],
            device=device,
            learning_rate=learning_rate,
            model_names=["actor_critic"])

        self.model_optimizer = self._get_optimizer(self.model.parameters())

        self.grad_clip_max = grad_clip_max
        self.batch_size = batch_size
        self.critic_loss_coefficient = critic_loss_coefficient
        self.gae_lambda = gae_lambda

        self.loss = None
        self.critic_loss = None
        self.actor_loss = None

        self.buffer = None

    def act(self, states: np.ndarray) -> (np.ndarray, np.ndarray):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        pred = self.model(states)
        return np.clip(pred["actions"].detach().cpu().numpy(), -1, 1), pred["actions"].detach().cpu().numpy(), \
               pred["dist"].log_prob(pred["actions"]).detach().cpu().numpy()

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

        shape = states.shape
        estimated_state_values = self.model(states.reshape(-1, shape[-1]))["v"].view(shape[0], shape[1]).detach()
        estimated_next_state_values = self.model(next_states.reshape(-1, shape[-1]))["v"].view(shape[0], shape[1]).detach()
        advantage, value_target = self.generalized_advantages_estimation(estimated_state_values=estimated_state_values,
                                                           estimated_next_state_values=estimated_next_state_values,
                                                           rewards=rewards)

        new_samples = [states.reshape(-1, states.shape[-1]), action_logits.reshape(-1, action_logits.shape[-1]),
                       action_log_probs.reshape(-1), value_target.reshape(-1), advantage.reshape(-1)]
        if self.buffer is None:
            self.buffer = new_samples
        else:
            self.buffer = [torch.cat((x, y), dim=0) for x, y in zip(self.buffer, new_samples)]

        buffer_length = self.buffer[0].shape[0]
        if buffer_length < self.batch_size * self.SGD_epoch:
            return

        states, action_logits, action_log_probs, value_target, advantage = self.buffer

        advantage = (advantage - advantage.mean()) / advantage.std()

        indices = torch.randperm(buffer_length)
        batches = [indices[i*self.batch_size:(i+1)*self.batch_size] for i, x in enumerate(range(self.SGD_epoch))]

        actor_losses = []
        critic_losses = []
        losses = []

        for minibatch_idx in batches:
            batch_states = states[minibatch_idx]
            batch_action_logits = action_logits[minibatch_idx]
            batch_action_log_probs = action_log_probs[minibatch_idx]
            batch_value_target = value_target[minibatch_idx]
            batch_advantage = advantage[minibatch_idx]

            pred = self.model(batch_states)
            batch_estimated_state_values = pred["v"].reshape(-1)
            critic_loss = self.critic_loss_coefficient * \
                               ((batch_value_target - batch_estimated_state_values) ** 2).mean()

            new_log_probs, entropy = pred["dist"].log_prob(batch_action_logits), pred["dist"].entropy()

            ratio = torch.exp(new_log_probs - batch_action_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            clipped_surrogate = torch.min(ratio * batch_advantage, clipped_ratio * batch_advantage)

            regularization = self.beta * entropy

            actor_loss = (clipped_surrogate + regularization).mean()
            loss = actor_loss + critic_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_max is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max)
            self.model_optimizer.step()

            actor_losses.append(actor_loss.detach().cpu().numpy())
            critic_losses.append(critic_loss.detach().cpu().numpy())
            losses.append(loss.detach().cpu().numpy())

        self.actor_loss = np.mean(actor_losses)
        self.critic_loss = np.mean(critic_losses)
        self.loss = np.mean(losses)

        # the clipping parameter reduces as time goes on
        self.epsilon *= self.epsilon_decay

        # the regulation term also reduces
        # this reduces exploration in later runs
        self.beta *= self.beta_deay

        self.buffer = None

    def generalized_advantages_estimation(self, estimated_state_values, estimated_next_state_values, rewards):
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

        value_target = advantage_estimation + estimated_state_values

        return advantage_estimation, value_target

    def get_discounted_future_rewards(self, rewards):
        """
        Calculate the discounted future rewards for each time step in each trajectory

        @param rewards: the raw received rewards [trajectories, time steps]
        @return: the discounted future rewards [trajectories, time steps]
        """

        T = rewards.shape[1]
        discounted_future_rewards = torch.empty_like(rewards)
        for t in range(T):
            coefficients = ((self.discount_rate) ** torch.arange(0, T - t, 1)).to(self.device)
            discounted_future_rewards[:, t] = (rewards[:, t:] * coefficients).sum(dim=1)

        return discounted_future_rewards

    def get_name(self) -> str:
        return "PPO_ActorCritic"

    def get_log_dict(self) -> dict:
        return {
            "var_mean": self.model.variance.detach().cpu().numpy().mean(),
            "beta": self.beta,
            "epsilon": self.epsilon,
            "critic_loss": self.critic_loss if self.critic_loss is not None else 0.0,
            "actor_loss": self.actor_loss if self.actor_loss is not None else 0.0,
            "loss": self.loss if self.loss is not None else 0.0,
            "grad_critic":
                np.array([x.grad.norm(dim=0).mean().detach().cpu().numpy() for x in self.model.parameters()]).mean()
                if self.loss is not None else 0.0,
            "grad_actor":
                np.array([x.grad.norm(dim=0).mean().detach().cpu().numpy() for x in self.model.parameters()]).mean()
                if self.loss is not None else 0.0
        }
