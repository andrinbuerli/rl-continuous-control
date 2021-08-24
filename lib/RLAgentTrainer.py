from collections import deque
import numpy as np
from typing import Union, List
import shutil
from glob import glob
import os

from lib.agent.BaseRLAgent import BaseRLAgent
from lib.env.ParallelAgentsBaseEnvironment import ParallelAgentsBaseEnvironment
from lib.log.BaseLogger import BaseLogger


class RLAgentTrainer:

    def __init__(
            self,
            agent: BaseRLAgent,
            env: ParallelAgentsBaseEnvironment,
            seed: int,
            logger: BaseLogger = None,
            agent_save_dir="agents"):
        """
        Initialize the RLAgentTrainer

        @param agent: the agent to be trained
        @param env:  the (continuous) environment to sample the trajectories from
        @param logger: the logger to monitor the training progress
        """

        self.agent_save_dir = agent_save_dir
        self.seed = seed
        self.logger = logger
        self.env = env
        self.agent = agent
        self.scores = []  # list containing scores from each episode
        self.scores_window = deque(maxlen=100)  # last 100 scores

        self.states = None
        self.trajectory_scores = None

        if not os.path.exists(self.agent_save_dir):
            os.mkdir(self.agent_save_dir)

    def train(
            self,
            n_iterations: int,
            max_t: Union[List[int], int],
            max_t_iteration:  Union[List[int], int] = None,
            intercept: bool = False):
        """
        RL agent training

        @param n_iterations: number of training iterations
        @param max_t: the number of time steps per sampled trajectory
        @param max_t_iteration: the iteration number at which the corresponding max_t should be used
        """

        max_t_original = max_t

        max_mean_score = None
        for i_iter in range(1, n_iterations + 1):
            if type(max_t_original) is list:
                max_t_index = min([i for i, tresh in enumerate(max_t_iteration) if tresh >= i_iter] + [len(max_t_original) - 1])
                max_t = max_t_original[max_t_index]

            states, actions, action_logits, log_probs, rewards, next_states = self.__collect_trajectories(
                max_t=max_t, intercept=intercept)
            print(f"{rewards.max()}\n")
            self.agent.learn(states=states, action_logits=action_logits, action_log_probs=log_probs,
                             rewards=rewards, next_states=next_states, actions=actions)

            score_window_mean = np.mean(self.scores_window)

            if max_mean_score is None or max_mean_score < score_window_mean:
                max_mean_score = score_window_mean
                directories = list(glob(f"{self.agent_save_dir}/*{self.seed}*"))
                if len(directories) > 0:
                    shutil.rmtree(directories[0])

                self.__save_agent(i_iter, score_window_mean)

            print('\rIteration {}\tAverage Score: {:.2f}'.format(i_iter, score_window_mean), end="")
            if i_iter % 100 == 0:
                print('\rIteration {}\tAverage Score: {:.2f}'.format(i_iter, score_window_mean))
            if score_window_mean >= self.env.target_reward:
                print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iter - 100,
                                                                                               score_window_mean))
                self.__save_agent(i_iter, score_window_mean)
                break

        score_window_mean = np.mean(self.scores_window)
        if score_window_mean < self.env.target_reward:
            print('\nEnvironment was not solved in {:d} iterations!\tAverage Score: {:.2f}'.format(n_iterations - 99,
                                                                                                   score_window_mean))
            self.__save_agent(i_iter, score_window_mean)

    def __save_agent(self, i_iter, score_window_mean):
        dir_name = f'{self.env.name}-{self.agent.get_name()}_{i_iter}-{self.seed}-{round(score_window_mean, 2)}'
        self.agent.save(directory_name=os.path.join(self.agent_save_dir, dir_name))

    def __collect_trajectories(self, max_t: int, intercept = False):
        """
        Sample trajectories from the environment

        @param max_t: max time steps
        @return: (
                states [trajectory, time_step, state_size], actions  [trajectory, time_step, action_size],
                action_logits  [trajectory, time_step, action_size], log_probs  [trajectory, time_step],
                rewards  [trajectory, time_step], next_states [trajectory, time_step, state_size]
            )
        """

        if not intercept or self.states is None:
            self.agent.reset()
            self.states = self.env.reset()
            self.trajectory_scores = np.zeros(self.env.num_agents)

        s_t0, a_t0, al_t0, pa_t0, r_t1, s_t1 = ([] for _ in range(6))

        t_sampled = None
        for t in range(max_t):
            actions, action_logits, log_probs = self.agent.act(self.states)
            next_states, rewards, dones = self.env.act(actions)

            t_sampled = t
            if any(dones):
                if intercept:
                    self.agent.reset()
                    self.states = self.env.reset()
                    self.trajectory_scores = np.zeros(self.env.num_agents)
                    continue
                else:
                    break

            s_t0.append(self.states), a_t0.append(actions), al_t0.append(action_logits), pa_t0.append(log_probs)
            r_t1.append(rewards), s_t1.append(next_states)

            self.states = next_states
            self.trajectory_scores += rewards

        mean_score = self.trajectory_scores.mean()

        if np.isnan(mean_score):
            print("!!!!! WARNING mean_score is NAN !!!!!")
            print(self.trajectory_scores)
            mean_score = self.scores_window[-1]

        self.scores_window.append(mean_score)  # save most recent score
        self.scores.append(mean_score)  # save most recent score

        if self.logger is not None:
            self.logger.log({
                "reward": mean_score,
                "trajectory_length": t_sampled,
                **self.agent.get_log_dict()
            })

        return np.transpose(np.array(s_t0), axes=[1, 0, 2]), np.transpose(np.array(a_t0), axes=[1, 0, 2]), \
               np.transpose(np.array(al_t0), axes=[1, 0, 2]), np.transpose(np.array(pa_t0), axes=[1, 0]), \
               np.transpose(np.array(r_t1), axes=[1, 0]), np.transpose(np.array(s_t1), axes=[1, 0, 2]),
