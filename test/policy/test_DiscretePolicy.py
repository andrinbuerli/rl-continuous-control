import numpy as np
from lib.policy.DiscretePolicy import DiscretePolicy

import torch


def test_forward_shape():
    testee = DiscretePolicy(state_size=10, action_size=2, seed=42)

    states = np.random.uniform(0, 1, (10, 10))

    actions, log_probs = testee.forward(torch.tensor(states))

    assert list(actions.shape) == [10] and list(log_probs.shape) == [10]


def test_forward_dtype():
    testee = DiscretePolicy(state_size=10, action_size=2, seed=42)

    states = np.random.uniform(0, 1, (10, 10))

    actions, log_probs = testee.forward(torch.tensor(states))

    assert actions.dtype == torch.int64