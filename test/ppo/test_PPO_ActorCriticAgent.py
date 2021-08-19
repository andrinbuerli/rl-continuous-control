import torch
import numpy as np
from lib.policy.ContinuousDiagonalGaussianPolicy import ContinuousDiagonalGaussianPolicy
from lib.policy.DiscretePolicy import DiscretePolicy
from lib.ppo.PPO_ActorCriticAgent import PPO_ActorCriticRLAgent
from lib.function.ValueFunction import ValueFunction
from test.ppo.MockPolicy import MockPolicy


def test_estimate_advantages_calculation():
    # arrange
    policy = MockPolicy(
        state_size=1, action_size=1, seed=42,
        return_forward_values=(
            torch.Tensor([[1], [1], [1]]), torch.log(torch.Tensor([0.89, 0.25, 0.3]))
        ))

    gamma = 0.9
    lambd = 0.9
    testee = PPO_ActorCriticRLAgent(
        beta=0,
        gae_lambda=lambd,
        actor=policy,
        discount_rate=gamma,
        critic=ValueFunction(state_size=1, seed=42)
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
                td_error[0, 0] + (gamma*lambd)*td_error[0, 1] + (gamma*lambd)**2*td_error[0, 2],
                td_error[0, 1] + (gamma*lambd)*td_error[0, 2],
                td_error[0, 2]
            ]
        ]
    )

    # act
    advantages = testee.estimate_advantages(estimated_state_values=torch.tensor(Vt_0),
                                            estimated_next_state_values=torch.tensor(Vt_1),
                                            rewards=torch.tensor(rewards)).detach()\
        .cpu().numpy()

    # assert
    assert np.isclose(gae_estimate, advantages).all()