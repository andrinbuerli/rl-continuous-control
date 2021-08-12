import torch
import numpy as np
from lib.policy.ContinuousDiagonalGaussianPolicy import ContinuousDiagonalGaussianPolicy
from lib.policy.DiscretePolicy import DiscretePolicy
from lib.ppo.PPORLAgent import PPORLAgent


def test_clipped_surrogate_function():
    # arrange
    policy = ContinuousDiagonalGaussianPolicy(state_size=1, action_size=1, seed=42)
    testee = PPORLAgent(
        policy=policy
    )

    timesteps = 50
    batch_size = 10

    old_probs = np.random.uniform(0, 1, (batch_size, timesteps))
    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    rewards = np.random.uniform(0, 1, (batch_size, timesteps))

    # act
    loss = testee._clipped_surrogate_function(
        old_probs=old_probs, states=states, rewards=rewards
    )

    # assert
    assert loss != 0 and not np.isnan(loss.detach().cpu().numpy())


def test_clipped_surrogate_function_backprop():
    # arrange
    policy = ContinuousDiagonalGaussianPolicy(state_size=2, action_size=2, seed=42)

    testee = PPORLAgent(
        policy=policy
    )

    timesteps = 50
    batch_size = 10

    old_probs = np.random.uniform(0, 1, (batch_size, timesteps))
    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    rewards = np.random.uniform(0, 1, (batch_size, timesteps))

    # act
    loss = testee._clipped_surrogate_function(
        old_probs=old_probs, states=states, rewards=rewards
    )

    loss.backward()

    # assert
    assert all([(x.grad != 0).detach().numpy().any() for x in policy.parameters()])


def test_act_continuous():
    # arrange
    policy = ContinuousDiagonalGaussianPolicy(state_size=2, action_size=2, seed=42)

    testee = PPORLAgent(
        policy=policy
    )

    timesteps = 50
    batch_size = 10

    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))

    # act
    actions = testee.act(states)

    # assert
    assert actions.shape == (batch_size, timesteps, policy.action_size) and actions.dtype == np.float32


def test_act_discrete():
    # arrange
    policy = DiscretePolicy(state_size=2, action_size=2, seed=42)

    testee = PPORLAgent(
        policy=policy
    )

    timesteps = 50
    batch_size = 10

    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))

    # act
    actions = testee.act(states)

    # assert
    assert actions.shape == (batch_size, timesteps) and actions.dtype == np.int


def test_learn():
    # arrange
    policy = ContinuousDiagonalGaussianPolicy(state_size=2, action_size=2, seed=42)

    testee = PPORLAgent(
        policy=policy,
        learning_rate=.1
    )

    timesteps = 50
    batch_size = 10

    probs = np.random.uniform(0, 1, (batch_size, timesteps))
    actions = np.random.uniform(0, 1, (batch_size, timesteps, policy.action_size))
    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    next_states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    rewards = np.random.uniform(0, 1, (batch_size, timesteps))
    dones = np.zeros((batch_size, timesteps))

    previous_policy_params = [x.detach().numpy().copy() for x in policy.parameters()]

    # act
    testee.learn(states=states, actions=actions, action_probs=probs, rewards=rewards,
                 next_states=next_states, dones=dones)

    # assert
    post_policy_params = [x.detach().numpy() for x in policy.parameters()]
    assert all([(x != y).any() for x, y in zip(previous_policy_params, post_policy_params)])


def test_learn_discrete():
    # arrange
    policy = DiscretePolicy(state_size=2, action_size=2, seed=42)

    testee = PPORLAgent(
        policy=policy,
        learning_rate=.1
    )

    timesteps = 50
    batch_size = 10

    probs = np.random.uniform(0, 1, (batch_size, timesteps))
    actions = np.random.uniform(0, 1, (batch_size, timesteps))
    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    next_states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    rewards = np.random.uniform(0, 1, (batch_size, timesteps))
    dones = np.zeros((batch_size, timesteps))

    previous_policy_params = [x.detach().numpy().copy() for x in policy.parameters()]

    # act
    testee.learn(states=states, actions=actions, action_probs=probs, rewards=rewards,
                 next_states=next_states, dones=dones)

    # assert
    post_policy_params = [x.detach().numpy() for x in policy.parameters()]
    assert all([(x != y).any() for x, y in zip(previous_policy_params, post_policy_params)])
