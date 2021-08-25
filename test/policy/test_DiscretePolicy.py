import numpy as np
from lib.policy.DiscretePolicy import DiscretePolicy

import torch


def test_forward_shape():
    testee = DiscretePolicy(state_size=10, action_size=2, seed=42)

    states = np.random.uniform(0, 1, (10, 10))

    actions, action_logits, dist = testee.forward(torch.tensor(states))

    assert list(actions.shape) == [10, 2] and list(action_logits.shape) == [10, 2]


def test_forward_dtype():
    testee = DiscretePolicy(state_size=10, action_size=2, seed=42)

    states = np.random.uniform(0, 1, (10, 10))

    actions, action_logits, dist = testee.forward(torch.tensor(states))

    assert actions.dtype == torch.float32 \
           and actions.min() == 0 and actions.max() == 1 # one-hot