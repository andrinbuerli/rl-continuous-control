from unityagents import UnityEnvironment

from lib.RLAgentTrainer import RLAgentTrainer
from lib.policy.ContinuousDiagonalGaussianPolicy import ContinuousDiagonalGaussianPolicy
from lib.ppo.PPORLAgent import PPORLAgent


if __name__ == "__main__":
    env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    trainer = RLAgentTrainer()

    policy = ContinuousDiagonalGaussianPolicy(state_size=state_size, action_size=action_size, seed=42,
                                              action_clip_range=[-1, 1])
    agent = PPORLAgent(policy=policy)

    trainer.train(10, agent=agent, env=env, max_t=100, log_wandb=False)

