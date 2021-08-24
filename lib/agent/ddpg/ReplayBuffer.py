import numpy as np
import random
from collections import namedtuple, deque

import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device="cpu"):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state):
        """Add a new experience to memory.

        Params
        ======
            state (np.array float): previous state
            action (int): executed action
            reward (float): collected reward
            next_state (np.array float): the next state
            priority (float): priority of experience (e.g. absolute temporal difference)
        """
        e = Experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)

        return (states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, peps=1e-3, seed=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            peps (float): constant term added to priority
            seed (int): random seed
        """
        self.peps = peps
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.memory.clear()
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, priority):
        """Add a new experience to memory.

        Params
        ======
            state (np.array float): previous state
            action (int): executed action
            reward (float): collected reward
            next_state (np.array float): the next state
            priority (float): priority of experience (e.g. absolute temporal difference)
        """
        e = Experience(state, action, reward, next_state, priority + self.peps)
        self.memory.append(e)

    def update_priorities(self, indices, priorities):
        """Update priorities in the memory at the given indices."""
        experiences = [self.memory.__getitem__(x) for x in indices]
        for i, exp in enumerate(experiences):
            exp.priority = float(priorities[i] + self.peps)

    def sample(self, a=1, b=1):
        """Sample a batch of experiences from memory according to the stored priorities.

        Params
        ======
            a (float): a=1 greedy prioritized sampling, a=0 uniform sampling
            b (float): b=1 fully accounted importance sampling weight, b=0 ignore importance sampling weight
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
        """Return the current size of internal memory."""
        return len(self.memory)


class Experience:
    def __init__(self, state, action, reward, next_state, priority=None):
        self.priority = priority
        self.next_state = next_state
        self.next_state = next_state
        self.reward = reward
        self.action = action
        self.state = state
