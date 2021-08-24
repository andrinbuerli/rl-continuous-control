import numpy as np
import torch
from typing import Callable
from random import  random

from lib.agent.BaseRLAgent import BaseRLAgent
from lib.policy import StochasticBasePolicy
from lib.function.StateActionValueFunction import StateActionValueFunction
from lib.agent.ddpg.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer

class AnnealedGaussianProcess:
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class DDPGRLAgent(BaseRLAgent):

    def __init__(
            self,
            get_actor,
            get_critic,
            state_size,
            action_size,
            seed=0,
            buffer_size=int(1e5),
            batch_size=64,
            gamma=0.99,
            tau=1e-3,
            lr=5e-4,
            update_every=4,
            update_for=4,
            double_dqn=False,
            dueling_networks=False,
            prioritized_exp_replay=False,
            prio_a=0.7,
            prio_b_init=0.5,
            prio_b_growth=1.1,
            epsilon=1,
            epsilon_decay=.9,
            epsilon_min=.01,
            device="cpu"
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters, θ_target = τ*θ_local + (1 - τ)*θ_target
            lr (float): learning rate
            update_every (int): how often to update the network, after every n step
            double_dqn (bool): use double Q learning with target network as secondary network
            prioritized_exp_replay (bool): use prioritized experience replay
            prio_a (float): a = 0 uniform sampling, a = 1 fully prioritized sampling
            prio_b_init (float): importance sampling weight init
            prio_b_gowth (float): importance sampling weight growth (will grow to max of 1)
        """
        self.epsilon_min = epsilon_min
        self.update_for = update_for
        self.eps = epsilon
        self.eps_decay = epsilon_decay
        self.state_size = state_size
        self.action_size = action_size
        self.prio_b_growth = prio_b_growth
        self.prio_b = prio_b_init
        self.prio_a = prio_a
        self.prioritized_exp_replay = prioritized_exp_replay
        self.dueling_networks = dueling_networks
        self.double_dqn = double_dqn
        self.update_every = update_every
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.qnetwork_local = get_critic().to(device)
        self.qnetwork_target = get_critic().to(device)

        self.argmaxpolicy_local = get_actor().to(device)
        self.argmaxpolicy_target = get_actor().to(device)

        self.random_process = OrnsteinUhlenbeckProcess(size=self.action_size)

        super(DDPGRLAgent, self).__init__(
            models=[self.qnetwork_local, self.qnetwork_target, self.argmaxpolicy_local, self.argmaxpolicy_target],
            device=device,
            learning_rate=lr,
            model_names=["qnetwork_local, qnetwork_target", "policy_local, policy_target"])

        self.qnetwork_optimizer = self._get_optimizer(self.qnetwork_local.parameters())
        self.argmaxpolicy_optimizer = self._get_optimizer(self.argmaxpolicy_local.parameters())

        # Replay memory
        if self.prioritized_exp_replay:
            self.memory = PrioritizedReplayBuffer(action_size, self.buffer_size, self.batch_size, seed=seed, peps=1e-4,
                                                  device=device)
        else:
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed=seed, device=device)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

        self.loss = None
        self.policy_gradients = None
        self.critic_loss = None

    def act(self, states: np.ndarray, training: int = 1) -> (np.ndarray, np.ndarray):
        """
        # Add random exploration noise
        if random() > self.eps:
            self.argmaxpolicy_local.eval()
            with torch.no_grad():
                actions = self.argmaxpolicy_local(states)
            self.argmaxpolicy_local.train()

            actions = actions.detach().cpu().numpy()
            return actions, np.zeros_like(actions), np.zeros((actions.shape[0]))
        else:
            return np.random.uniform(-1, 1, (states.shape[0], self.action_size)),\
                   np.zeros((states.shape[0], self.action_size)), np.zeros((states.shape[0]))
        """

        states = torch.from_numpy(states).float().to(self.device)

        self.argmaxpolicy_local.eval()
        with torch.no_grad():
            actions = self.argmaxpolicy_local(states)
        self.argmaxpolicy_local.train()

        actions = actions.detach().cpu().numpy()

        actions += training * max(self.eps, 0) * self.random_process.sample()
        actions = np.clip(actions, -1., 1.)

        return actions, np.zeros_like(actions), np.zeros((actions.shape[0]))


    def learn(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            action_logits: np.ndarray,
            action_log_probs: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray):
        # Save experience in replay memory
        for state, action, reward, next_state in zip(states.reshape(states.shape[0] * states.shape[1], -1),
                                                     actions.reshape(states.shape[0] * states.shape[1], -1),
                                                     rewards.reshape(states.shape[0] * states.shape[1], -1),
                                                     next_states.reshape(states.shape[0] * states.shape[1], -1)):
            if self.prioritized_exp_replay:
                priority = np.abs(self.calculate_td_error(
                    torch.from_numpy(state).float().to(device),
                    torch.tensor(action).float().to(device),
                    torch.tensor(reward).float().to(device),
                    torch.from_numpy(next_state).float().to(device),
                    torch.tensor(done).float().to(device)).cpu().detach().numpy()[0, 0])

                self.memory.add(state, action, reward, next_state, priority)
            else:
                self.memory.add(state, action, reward, next_state)

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            for _ in range(self.update_for):
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > self.batch_size:
                    if self.prioritized_exp_replay:
                        experiences = self.memory.sample(a=self.prio_a, b=self.prio_b)
                        self.prio_b = min(1, self.prio_b + self.prio_b_growth)
                    else:
                        experiences = self.memory.sample()

                    self.__learn(experiences)

            self.eps = max(self.eps * self.eps_decay, self.epsilon_min)

    def get_name(self) -> str:
        return "DDPG"

    def __learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """

        if self.prioritized_exp_replay:
            states, actions, rewards, next_states, indices, importance_sampling_weights = experiences
        else:
            states, actions, rewards, next_states = experiences

        td_error = self.calculate_td_error(states, actions, rewards, next_states)

        if self.prioritized_exp_replay:
            self.memory.update_priorities(indices, np.abs(td_error.cpu().detach().numpy()))

        if self.prioritized_exp_replay:
            self.critic_loss = (td_error ** 2 * importance_sampling_weights).mean()
        else:
            self.critic_loss = (td_error ** 2).mean()

        actions = self.argmaxpolicy_local(states)
        self.policy_gradients = -(self.qnetwork_local(states, actions)).mean()

        self.loss = self.critic_loss + self.policy_gradients

        self.argmaxpolicy_optimizer.zero_grad()
        self.qnetwork_optimizer.zero_grad()
        self.loss.backward()
        self.argmaxpolicy_optimizer.step()
        self.qnetwork_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        self.soft_update(self.argmaxpolicy_local, self.argmaxpolicy_target)

    def calculate_td_error(self, states, actions, rewards, next_states):
        """Calculate temporal difference error

        Params
        ======
            states (PyTorch float tensor): previous states
            actions (PyTorch int tensor): executed actions
            rewards (PyTorch float tensor): collected rewards
            next_states (PyTorch float tensor): the next states
        """

        next_best_actions = self.argmaxpolicy_target.forward(next_states)
        q_values_next_state = self.qnetwork_target.forward(next_states, next_best_actions)

        q_targets = rewards + self.gamma * q_values_next_state
        q_values = self.qnetwork_local.forward(states, actions)
        td_error = q_values - q_targets

        return td_error

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def get_log_dict(self) -> dict:
        return {
            "epsilon": self.eps,
            "critic_loss": self.critic_loss.detach().cpu().numpy() if self.critic_loss is not None else None,
            "actor_loss": self.policy_gradients.detach().cpu().numpy() if self.policy_gradients is not None else None,
            "loss": self.loss.detach().cpu().numpy() if self.loss is not None else None,
            "grad_actor":
                np.array([x.grad.norm(dim=0).mean().detach().cpu().numpy() for x in self.argmaxpolicy_local.parameters()]).mean()
                if self.policy_gradients is not None else None,
            "grad_critic":
                np.array([x.grad.norm(dim=0).mean().detach().cpu().numpy() for x in self.qnetwork_local.parameters()]).mean()
                if self.critic_loss is not None else None
        }
