import numpy as np

from lib.agent.ddpg.ReplayBuffer import PrioritizedReplayBuffer


def test_add():
    # arrange
    buffer = PrioritizedReplayBuffer(action_size=1, buffer_size=1, batch_size=1, seed=0)
    prev_len = len(buffer)

    # act
    buffer.add(state=1, action=2, reward=3, next_state=4, priority=1)

    # assert
    assert prev_len == 0 and len(buffer) == 1


def test_sample():
    # arrange
    batch_size = 10
    buffer = PrioritizedReplayBuffer(action_size=1, buffer_size=30, batch_size=batch_size, seed=0)

    [buffer.add(state=1, action=2, reward=3, next_state=4, priority=1) for _ in range(20)]

    # act
    (states, actions, rewards, next_states, indices, importance_sampling_weight) = buffer.sample()

    # assert
    assert len(states) == batch_size and len(actions) == batch_size and len(rewards) == batch_size \
           and len(next_states) == batch_size and len(indices) == batch_size \
           and len(importance_sampling_weight) == batch_size


def test_priority_sample():
    # arrange
    batch_size = 10
    buffer = PrioritizedReplayBuffer(action_size=1, buffer_size=30, batch_size=batch_size, seed=0)

    [buffer.add(state=i, action=2, reward=3, next_state=4, priority=i) for i in range(20)]

    # act
    for _ in range(100):
        sampled_states = []
        for _ in range(100):
            (states, actions, rewards, next_states, indices,
             importance_sampling_weight) = buffer.sample()
            sampled_states.extend(states.numpy().reshape(-1))

        # assert
        values, bins = np.histogram(sampled_states, bins=range(20))
        difference = np.diff(values)

        # number of samples per state should in general be increasing with the
        # state number because the priorities are as well
        assert difference.mean() > 0
