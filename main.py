import torch

from lib.RLAgentTrainer import RLAgentTrainer
from lib.env.ParallelAgentsUnityEnvironment import ParallelAgentsUnityEnvironment
from lib.policy.ContinuousDiagonalGaussianPolicy import ContinuousDiagonalGaussianPolicy
from lib.ppo.PPORLAgent import PPORLAgent

if __name__ == "__main__":
    env = ParallelAgentsUnityEnvironment(env_binary_path='Reacher_Linux_NoVis/Reacher.x86_64')

    policy = ContinuousDiagonalGaussianPolicy(state_size=env.state_size, action_size=env.action_size, seed=42,
                                              output_transform=lambda x: torch.tanh(x))
    agent = PPORLAgent(policy=policy, learning_rate=1e-4)

    trainer = RLAgentTrainer(agent=agent, env=env, log_wandb=False)
    trainer.train(n_iterations=1000, max_t=100)

