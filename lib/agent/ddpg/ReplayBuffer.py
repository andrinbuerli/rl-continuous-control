import numpy as np
import random
from collections import namedtuple, deque

import torch


class ReplayBuffer:

    def __init__(
            self,
            action_size: int,
            buffer_size: int,
            batch_size: int,
            seed: int,
            device: str = "cpu"):
        """
        Fixed-size buffer to store experience tuples.

        @param action_size: dimension of each action
        @param buffer_size: maximum size of buffer
        @param batch_size: size of each training batch
        @param seed: random seed
        @param device: device for calculations
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = seed
        random.seed(seed)

    def add(self, state: np.array, action: np.array, reward: float, next_state: np.array):
        """
        Add a new experience to memory.

        @param state: previous state
        @param action: executed action
        @param reward: collected reward
        @param next_state: the next state
        """
        e = __Experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.

        @return: (states, actions, rewards, next_states)
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)

        return states, actions, rewards, next_states

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer:

    def __init__(
            self,
            action_size: int,
            buffer_size: int,
            batch_size: int,
            peps: float = 1e-3,
            seed: int = 0,
            device: str = "cpu"):
        """
        Fixed-size buffer to store experience tuples.



        @param action_size: dimension of each action
        @param buffer_size: maximum size of buffer
        @param batch_size: size of each training batch
        @param peps: constant term added to priority
        @param seed: random seed
        @param device: device for calculations
        """
        self.device = device
        self.peps = peps
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.memory.clear()
        self.batch_size = batch_size
        self.seed = seed
        random.seed(seed)

    def add(self, state: np.array, action: np.array, reward: float, next_state: np.array, priority: float):
        """
        Add a new experience to memory.

        @param state: previous state
        @param action: executed action
        @param reward: collected reward
        @param next_state: the next state
        @param priority: priority of experience (e.g. absolute temporal difference)
        """
        e = __Experience(state, action, reward, next_state, priority + self.peps)
        self.memory.append(e)

    def update_priorities(self, indices, priorities):
        """
        Update priorities in the memory at the given indices.
        """
        experiences = [self.memory.__getitem__(x) for x in indices]
        for i, exp in enumerate(experiences):
            exp.priority = float(priorities[i] + self.peps)

    def sample(self, a: float = 1, b: float = 1):
        """
        Sample a batch of experiences from memory according to the stored priorities.

        @param a: a=1 greedy prioritized sampling, a=0 uniform sampling
        @param b: b=1 fully accounted importance sampling weight, b=0 ignore importance sampling weight
        @return: (states, actions, rewards, next_states)
        """

        priorities = np.array(list(map(lambda exp: exp.priority, self.memory)))
        powered = priorities**a

        props = powered / powered.sum()

        indices = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, p=props)
        experiences = [self.memory.__getitem__(x) for x in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        priorities = torch.from_numpy(props[indices]).float().to(
            self.device)

        importance_sampling_weight = (len(self.memory) * priorities) ** (-b)
        importance_sampling_weight /= importance_sampling_weight.max()

        return states, actions, rewards, next_states, indices, importance_sampling_weight

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)


class __Experience:
    def __init__(self, state: np.array, action: np.array, reward: float, next_state: np.array, priority=None):
        self.priority = priority
        self.next_state = next_state
        self.next_state = next_state
        self.reward = reward
        self.action = action
        self.state = state
