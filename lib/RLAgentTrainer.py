from collections import deque
import numpy as np

from lib.BaseRLAgent import BaseRLAgent
from lib.env.ParallelAgentsBaseEnvironment import ParallelAgentsBaseEnvironment


class RLAgentTrainer:

    def __init__(
            self,
            agent: BaseRLAgent,
            env: ParallelAgentsBaseEnvironment,
            log_wandb: bool = False):
        self.env = env
        self.agent = agent
        self.log_wandb = log_wandb
        self.scores = []  # list containing scores from each episode
        self.scores_window = deque(maxlen=100)  # last 100 scores

    def train(
            self,
            n_iterations: int,
            max_t: int):
        """RL agent training

        Params
        ======
            n_episodes (int): maximum number of training episodes
        """

        for i_iter in range(1, n_iterations + 1):
            states, actions, probs, rewards, next_states, dones = self.__collect_trajectories(max_t=max_t)
            self.agent.learn(states, actions, probs, rewards, next_states, dones)

            score_window_mean = np.mean(self.scores_window)
            print('\rIteration {}\tAverage Score: {:.2f}'.format(i_iter, score_window_mean), end="")
            if i_iter % 100 == 0:
                print('\rIteration {}\tAverage Score: {:.2f}'.format(i_iter, score_window_mean))
            if score_window_mean >= 18.0:
                print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iter - 100,
                                                                                               score_window_mean))
                agent.save(directory_name=f'{agent.get_name()}_{i_iter - 100}-{round(score_window_mean, 2)}')
                break

        score_window_mean = np.mean(self.scores_window)
        print('\nEnvironment was not solved in {:d} iterations!\tAverage Score: {:.2f}'.format(n_iterations - 99,
                                                                                               score_window_mean))
        agent.save(directory_name=f'{agent.get_name()}_{n_iterations - 99}-{round(score_window_mean, 2)}')

    def __collect_trajectories(self, max_t: int):
        """

        @param max_t: max time steps
        @return: trajectories [nr_agents, trajectory]
        """
        states = self.env.reset()
        scores = np.zeros(self.env.num_agents)

        s_t0, a_t0, pa_t0, r_t1, s_t1, dones_t1 = ([] for _ in range(6))
        for t in range(max_t):
            actions, probs = self.agent.act(states)
            next_states, rewards, dones = self.env.act(actions)

            s_t0.append(states), a_t0.append(actions), pa_t0.append(probs)
            r_t1.append(rewards), s_t1.append(next_states), dones_t1.append(dones)

            states = next_states
            scores += rewards

        self.scores_window.append(scores.mean())  # save most recent score
        self.scores.append(scores.mean())  # save most recent score

        if self.log_wandb:
            wandb.log({
                "score": self.scores,
                "avg. score": self.scores_window
            })

        return np.transpose(np.array(s_t0), axes=[1, 0, 2]), np.transpose(np.array(a_t0), axes=[1, 0, 2]), \
               np.transpose(np.array(pa_t0), axes=[1, 0]), np.transpose(np.array(r_t1), axes=[1, 0]), \
               np.transpose(np.array(s_t1), axes=[1, 0, 2]), np.transpose(np.array(dones_t1), axes=[1, 0])
