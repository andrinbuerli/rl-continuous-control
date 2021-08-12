import numpy as np
import abc
import torch
import torch.nn as nn
import os


class BaseRLAgent:

    def __init__(
            self,
            models: [nn.Module],
            device: str,
            learning_rate: float):
        self.learning_rate = learning_rate
        self.models = models
        self.device = device

    def _get_optimizer(self, params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=self.learning_rate)

    @abc.abstractmethod
    def act(self, states: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def learn(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            action_probs: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            dones: np.ndarray):
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        pass

    def load(self, directory_name: str):
        if not os.path.exists(directory_name):
            raise FileNotFoundError(f"Directory {directory_name} not found")

        for model in self.models:
            model.load_state_dict(torch.load(os.path.join(directory_name, model.network_name + ".pth")))

    def save(self, directory_name: str):
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)

        for model in self.models:
            model.save(os.path.join(directory_name, model.network_name + ".pth"))
