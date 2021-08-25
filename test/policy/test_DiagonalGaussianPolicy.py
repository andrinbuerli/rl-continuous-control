import numpy as np
from lib.policy.StochasticContinuousGaussianPolicy import StochasticContinuousGaussianPolicy

import torch


def test_forward_shape():
    testee = StochasticContinuousGaussianPolicy(state_size=10, action_size=2, seed=42)

    states = np.random.uniform(0, 1, (10, 10))

    actions, action_logits, dist = testee.forward(torch.tensor(states))

    assert list(actions.shape) == [10, 2] and list(action_logits.shape) == [10, 2]


def test_forward_type():
    testee = StochasticContinuousGaussianPolicy(state_size=10, action_size=2, seed=42)

    states = np.random.uniform(0, 1, (10, 10))

    actions, action_logits, dist = testee.forward(torch.tensor(states))

    assert actions.dtype == torch.float32

