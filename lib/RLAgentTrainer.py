from collections import deque
import numpy as np

from lib.BaseRLAgent import BaseRLAgent
from lib.env.ParallelAgentsBaseEnvironment import ParallelAgentsBaseEnvironment
from lib.log.BaseLogger import BaseLogger


class RLAgentTrainer:

    def __init__(
            self,
            agent: BaseRLAgent,
            env: ParallelAgentsBaseEnvironment,
            logger: BaseLogger = None):
        self.logger = logger
        self.env = env
        self.agent = agent
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
            states, actions, action_logits, log_probs, rewards, next_states, dones = self.__collect_trajectories(max_t=max_t)
            print(f"{rewards.max()}\n")
            self.agent.learn(states=states, action_logits=action_logits, action_log_probs=log_probs,
                             rewards=rewards, next_states=next_states, dones=dones)

            score_window_mean = np.mean(self.scores_window)
            print('\rIteration {}\tAverage Score: {:.2f}'.format(i_iter, score_window_mean), end="")
            if i_iter % 100 == 0:
                print('\rIteration {}\tAverage Score: {:.2f}'.format(i_iter, score_window_mean))
            if score_window_mean >= 35.0:
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

        s_t0, a_t0, al_t0, pa_t0, r_t1, s_t1, dones_t1 = ([] for _ in range(7))
        for t in range(max_t):
            actions, action_logits, log_probs = self.agent.act(states)
            next_states, rewards, dones = self.env.act(actions)

            s_t0.append(states), a_t0.append(actions), al_t0.append(action_logits), pa_t0.append(log_probs)
            r_t1.append(rewards), s_t1.append(next_states), dones_t1.append(dones)

            states = next_states
            scores += rewards

        mean_score = scores.mean()
        self.scores_window.append(mean_score)  # save most recent score
        self.scores.append(mean_score)  # save most recent score

        if self.logger is not None:
            self.logger.log({
                "reward": mean_score,
            })

        return np.transpose(np.array(s_t0), axes=[1, 0, 2]), \
               np.transpose(np.array(a_t0), axes=[1, 0, 2]), np.transpose(np.array(al_t0), axes=[1, 0, 2]), \
               np.transpose(np.array(pa_t0), axes=[1, 0]), np.transpose(np.array(r_t1), axes=[1, 0]), \
               np.transpose(np.array(s_t1), axes=[1, 0, 2]), np.transpose(np.array(dones_t1), axes=[1, 0])
