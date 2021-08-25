import torch
import numpy as np
from lib.policy.StochasticContinuousGaussianPolicy import StochasticContinuousGaussianPolicy
from lib.policy.StochasticDiscretePolicy import StochasticDiscretePolicy
from lib.agent.ppo.PPORLAgent import PPORLAgent
from test.ppo.MockPolicy import MockPolicy


def test_clipped_surrogate_function():
    # arrange
    policy = StochasticContinuousGaussianPolicy(state_size=1, action_size=1, seed=42)
    testee = PPORLAgent(
        policy=policy
    )

    timesteps = 50
    batch_size = 10

    old_probs = torch.from_numpy(np.random.uniform(0, 1, (batch_size, timesteps))).to(torch.float32)
    action_logits = torch.from_numpy(np.random.uniform(0, 1, (batch_size, timesteps, policy.action_size))).to(
        torch.float32)
    states = torch.from_numpy(np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))).to(torch.float32)
    rewards = torch.from_numpy(np.random.uniform(0, 1, (batch_size, timesteps))).to(torch.float32)

    # act
    loss = testee.clipped_surrogate_function(
        old_log_probs=old_probs, states=states,
        future_discounted_rewards=rewards, action_logits=action_logits
    )

    # assert
    assert loss != 0 and not np.isnan(loss.detach().cpu().numpy())


def test_clipped_surrogate_calculation():
    # arrange
    dist = torch.distributions.MultivariateNormal(
        loc=torch.zeros([3, 2]),
        covariance_matrix=torch.diag_embed(torch.ones(3, 2)))
    policy = MockPolicy(
        state_size=1, action_size=2, seed=42,
        return_forward_values=(
            torch.Tensor([[1, 1], [1, 1], [1, 1]]),
            torch.log(torch.Tensor([0.89, 0.25, 0.3])),
            dist
        ))

    testee = PPORLAgent(
        beta=0,
        policy=policy
    )

    old_log_probs = np.log(np.array([[0.1, 0.2, 0.4]]))
    states = np.array([[[10], [11], [5]]])
    action_logits = np.array([[[10, 10], [11, 11], [5, 5]]])
    rewards = np.array([[0, 1, 2]])

    discounted_rewards = rewards * np.array(
        [testee.discount_rate ** 0, testee.discount_rate ** 1, testee.discount_rate ** 2])
    rewards_future = discounted_rewards.reshape(-1)[::-1].cumsum()[::-1].reshape(1, -1)
    log_new_probs = dist.log_prob(torch.from_numpy(action_logits.reshape(-1, action_logits.shape[-1]))).reshape(1, 3, 1)
    policy.seed = torch.manual_seed(42)

    old_log_probs, states, action_logits, rewards = [torch.from_numpy(x).to(torch.float32) for x in
                                                     [old_log_probs, states,
                                                      action_logits, rewards]]

    # act
    loss = testee.clipped_surrogate_function(
        old_log_probs=old_log_probs, states=states,
        future_discounted_rewards=rewards, action_logits=action_logits
    ).detach().cpu().numpy()

    # assert
    predicted_loss = np.array([min(
        torch.exp(log_new_probs[0][i] - old_log_probs[0][i]) * rewards_future[0][i],
        np.clip(torch.exp(log_new_probs[0][i] - old_log_probs[0][i]), 1 - testee.epsilon, 1 + testee.epsilon) *
        rewards_future[0][i]
    ) for i in range(3)]).mean()

    assert np.isclose(predicted_loss, loss, atol=1e-8)


def test_clipped_surrogate_function_backprop():
    # arrange
    policy = StochasticContinuousGaussianPolicy(state_size=2, action_size=2, seed=42)

    testee = PPORLAgent(
        policy=policy
    )

    timesteps = 50
    batch_size = 10

    old_log_probs = np.random.uniform(0, 1, (batch_size, timesteps))
    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    action_logits = np.random.uniform(0, 1, (batch_size, timesteps, policy.action_size))
    rewards = np.random.uniform(0, 1, (batch_size, timesteps))
    old_log_probs, states, action_logits, rewards = [torch.from_numpy(x).to(torch.float32) for x in
                                                     [old_log_probs, states,
                                                      action_logits, rewards]]

    # act
    loss = testee.clipped_surrogate_function(
        old_log_probs=old_log_probs, states=states,
        action_logits=action_logits, future_discounted_rewards=rewards
    )

    loss.backward()

    # assert
    assert all([(x.grad != 0).detach().cpu().numpy().any() for x in policy.parameters()])


def test_act_continuous():
    # arrange
    policy = StochasticContinuousGaussianPolicy(state_size=2, action_size=2, seed=42)

    testee = PPORLAgent(
        policy=policy
    )

    timesteps = 50
    batch_size = 10

    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))

    # act
    actions, _, _ = testee.act(states)

    # assert
    assert actions.shape == (batch_size, timesteps, policy.action_size) and actions.dtype == np.float32


def test_act_discrete():
    # arrange
    policy = StochasticDiscretePolicy(state_size=2, action_size=2, seed=42)

    testee = PPORLAgent(
        policy=policy
    )

    timesteps = 50
    batch_size = 10

    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))

    # act
    actions, _, _ = testee.act(states)

    # assert
    assert actions.shape == (batch_size, timesteps, policy.action_size) and actions.dtype == np.float32\
           and actions.min() == 0 and actions.max() == 1  # one-hot


def test_learn():
    # arrange
    policy = StochasticContinuousGaussianPolicy(state_size=2, action_size=2, seed=42)

    testee = PPORLAgent(
        policy=policy,
        learning_rate=.1
    )

    timesteps = 50
    batch_size = 10

    probs = np.random.uniform(0, 1, (batch_size, timesteps))
    actions = np.random.uniform(0, 1, (batch_size, timesteps, policy.action_size))
    action_logits = np.random.uniform(0, 1, (batch_size, timesteps, policy.action_size))
    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    next_states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    rewards = np.random.uniform(0, 1, (batch_size, timesteps))

    previous_policy_params = [x.detach().cpu().numpy().copy() for x in policy.parameters()]

    # act
    testee.learn(states=states, actions=actions, action_log_probs=probs, rewards=rewards,
                 next_states=next_states, action_logits=action_logits)

    # assert
    post_policy_params = [x.detach().cpu().numpy() for x in policy.parameters()]
    assert all([(x != y).any() for x, y in zip(previous_policy_params, post_policy_params)])


def test_learn_discrete():
    # arrange
    policy = StochasticDiscretePolicy(state_size=2, action_size=2, seed=42)

    testee = PPORLAgent(
        policy=policy,
        learning_rate=.1
    )

    timesteps = 50
    batch_size = 10

    probs = np.random.uniform(0, 1, (batch_size, timesteps))
    actions = np.random.uniform(0, 1, (batch_size, timesteps))
    action_logits = np.random.choice([0, 1], (batch_size, timesteps))
    action_logits = torch.nn.functional.one_hot(torch.from_numpy(action_logits).to(torch.int64), num_classes=2)
    states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    next_states = np.random.uniform(0, 1, (batch_size, timesteps, policy.state_size))
    rewards = np.random.uniform(0, 1, (batch_size, timesteps))

    previous_policy_params = [x.detach().cpu().numpy().copy() for x in policy.parameters()]

    # act
    testee.learn(states=states, actions=actions, action_log_probs=probs, rewards=rewards,
                 next_states=next_states, action_logits=action_logits)

    # assert
    post_policy_params = [x.detach().cpu().numpy() for x in policy.parameters()]
    assert all([(x != y).any() for x, y in zip(previous_policy_params, post_policy_params)])
