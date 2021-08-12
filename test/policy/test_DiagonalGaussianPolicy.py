import numpy as np
from lib.policy.ContinuousDiagonalGaussianPolicy import ContinuousDiagonalGaussianPolicy

import torch


def test_forward_shape():
    testee = ContinuousDiagonalGaussianPolicy(state_size=10, action_size=2, seed=42)

    states = np.random.uniform(0, 1, (10, 10))

    actions, log_probs = testee.forward(torch.tensor(states))

    assert list(actions.shape) == [10, 2] and list(log_probs.shape) == [10]


def test_forward_type():
    testee = ContinuousDiagonalGaussianPolicy(state_size=10, action_size=2, seed=42)

    states = np.random.uniform(0, 1, (10, 10))

    actions, log_probs = testee.forward(torch.tensor(states))

    assert actions.dtype == torch.float32

