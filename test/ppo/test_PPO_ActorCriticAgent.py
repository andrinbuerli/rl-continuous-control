import torch
import numpy as np
from lib.agent.ppo.PPOActorCriticRLAgent import PPOActorCriticRLAgent
from lib.models.function import StateValueFunction
from test.ppo.MockPolicy import MockPolicy


def test_estimate_advantages_calculation():
    # arrange
    dist = torch.distributions.Normal(torch.zeros(3, 1), torch.ones(3, 1))
    policy = MockPolicy(
        state_size=1, action_size=1, seed=42,
        return_forward_values=
        {
            "actions": torch.Tensor([[1], [1], [1]]),
            "actions_mode": torch.Tensor([[1], [1], [1]]),
            "dist": dist
        })

    gamma = 0.9
    lambd = 0.9
    testee = PPOActorCriticRLAgent(
        beta=0,
        gae_lambda=lambd,
        model=policy,
        discount_rate=gamma
    )

    rewards = np.array([[-1, 1, 2]])

    policy.seed = torch.manual_seed(42)

    Vt_0 = np.array([[3, -1, 2]])
    Vt_1 = np.array([[-1, 2, 4]])
    td_error = np.array(
        [
            [
                rewards[0, 0] + gamma * Vt_1[0, 0] - Vt_0[0, 0],
                rewards[0, 1] + gamma * Vt_1[0, 1] - Vt_0[0, 1],
                rewards[0, 2] + gamma * Vt_1[0, 2] - Vt_0[0, 2],
            ]
        ]
    )
    gae_estimate = np.array(
        [
            [
                td_error[0, 0] + (gamma * lambd) * td_error[0, 1] + (gamma * lambd) ** 2 * td_error[0, 2],
                td_error[0, 1] + (gamma * lambd) * td_error[0, 2],
                td_error[0, 2]
            ]
        ]
    )

    # act
    advantages = testee.generalized_advantages_estimation(estimated_state_values=torch.tensor(Vt_0),
                                                          estimated_next_state_values=torch.tensor(Vt_1),
                                                          rewards=torch.tensor(rewards))

    # assert
    assert np.isclose(gae_estimate, advantages.detach().cpu().numpy()).all()


def test_estimate_advantages_calculation_recover_td():
    # arrange
    dist = torch.distributions.Normal(torch.zeros(3, 1), torch.ones(3, 1))
    policy = MockPolicy(
        state_size=1, action_size=1, seed=42,
        return_forward_values=
        {
            "actions": torch.Tensor([[1], [1], [1]]),
            "actions_mode": torch.Tensor([[1], [1], [1]]),
            "dist": dist
        })

    gamma = 0.9
    lambd = 0
    testee = PPOActorCriticRLAgent(
        beta=0,
        gae_lambda=lambd,
        model=policy,
        discount_rate=gamma
    )

    rewards = np.array([[-1, 1, 2]])

    policy.seed = torch.manual_seed(42)

    Vt_0 = np.array([[3, -1, 2]])
    Vt_1 = np.array([[-1, 2, 4]])
    td_error = np.array(
        [
            [
                rewards[0, 0] + gamma * Vt_1[0, 0] - Vt_0[0, 0],
                rewards[0, 1] + gamma * Vt_1[0, 1] - Vt_0[0, 1],
                rewards[0, 2] + gamma * Vt_1[0, 2] - Vt_0[0, 2],
            ]
        ]
    )
    gae_estimate = np.array(
        [
            [
                td_error[0, 0],
                td_error[0, 1],
                td_error[0, 2]
            ]
        ]
    )

    # act
    advantages = testee.generalized_advantages_estimation(estimated_state_values=torch.tensor(Vt_0),
                                                          estimated_next_state_values=torch.tensor(Vt_1),
                                                          rewards=torch.tensor(rewards))

    # assert
    assert np.isclose(gae_estimate, advantages.detach().cpu().numpy()).all()


def test_estimate_advantages_calculation_recover_mc():
    # arrange
    dist = torch.distributions.Normal(torch.zeros(3, 1), torch.ones(3, 1))
    policy = MockPolicy(
        state_size=1, action_size=1, seed=42,
        return_forward_values=
        {
            "actions": torch.Tensor([[1], [1], [1]]),
            "actions_mode": torch.Tensor([[1], [1], [1]]),
            "dist": dist
        })

    gamma = 1
    lambd = 1
    testee = PPOActorCriticRLAgent(
        beta=0,
        gae_lambda=lambd,
        model=policy,
        discount_rate=gamma
    )

    rewards = np.array([[-1, 1, 2]])

    policy.seed = torch.manual_seed(42)

    Vt_0 = np.array([[3, -1, 2]])
    Vt_1 = np.array([[-1, 2, 4]])

    gae_estimate = np.array(
        [
            [
                rewards[0, 0] + rewards[0, 1] + rewards[0, 2] + Vt_1[0, 2] - Vt_0[0, 0],
                rewards[0, 1] + rewards[0, 2] + Vt_1[0, 2] - Vt_0[0, 1],
                rewards[0, 2] + Vt_1[0, 2] - Vt_0[0, 2],
            ]
        ]
    )

    # act
    advantages = testee.generalized_advantages_estimation(estimated_state_values=torch.tensor(Vt_0),
                                                          estimated_next_state_values=torch.tensor(Vt_1),
                                                          rewards=torch.tensor(rewards))

    # assert
    assert np.isclose(gae_estimate, advantages.detach().cpu().numpy()).all()
