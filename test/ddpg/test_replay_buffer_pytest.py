from lib.agent.ddpg.ReplayBuffer import ReplayBuffer


def test_add():
    # arrange
    buffer = ReplayBuffer(action_size=1, buffer_size=1, batch_size=1, seed=0)
    prev_len = len(buffer)

    # act
    buffer.add(state=1, action=2, reward=3, next_state=4)

    # assert
    assert prev_len == 0 and len(buffer) == 1


def test_sample():
    # arrange
    batch_size = 10
    buffer = ReplayBuffer(action_size=1, buffer_size=30, batch_size=batch_size, seed=0)

    [buffer.add(state=1, action=2, reward=3, next_state=4) for _ in range(20)]

    # act
    (states, actions, rewards, next_states) = buffer.sample()

    # assert
    assert len(states) == batch_size and len(actions) == batch_size and len(rewards) == batch_size \
           and len(next_states) == batch_size
