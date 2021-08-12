from lib.BaseRLAgent import BaseRLAgent


class RLAgentTrainer:

    def __init__(self):
        self.scores = []  # list containing scores from each episode
        self.scores_window = deque(maxlen=100)  # last 100 scores

    def train(
            self,
            n_iterations: int,
            agent: BaseRLAgent,
            env: UnityEnvironment,
            max_t: int,
            log_wandb: bool = True):
        """RL agent training

        Params
        ======
            n_episodes (int): maximum number of training episodes
        """

        for i_iter in range(1, n_iterations + 1):

            env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
            states = env_info.vector_observations  # get the current state

            scores = np.zeros(env_info.agents)
            for t in range(max_t):
                actions = agent.act(states)
                env_info = env.step(actions)[brain_name]  # send the action to the environment
                next_states = env_info.vector_observations  # get next state (for each agent)
                rewards = env_info.rewards  # get the reward
                dones = env_info.local_done  # see if episode has finished

                agent.learn(states, actions, rewards, next_states, dones)
                states = next_states
                scores += rewards
                if done:
                    break

            self.scores_window.append(scores.mean())  # save most recent score
            self.scores.append(scores.mean())  # save most recent score

            if log_wandb is not None:
                wandb.log({
                    "score": self.scores,
                    "avg. score": self.scores_window
                })

            print('\rIteration {}\tAverage Score: {:.2f}'.format(i_episode, self.scores_window.mean()), end="")
            if i_episode % 100 == 0:
                print('\rIteration {}\tAverage Score: {:.2f}'.format(i_episode, self.scores_window.mean()))
            if np.mean(scores_window) >= 18.0:
                print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iter - 100,
                                                                                               self.scores_window.mean()))
                agent.save(directory_name=f'{agent.get_name()}_{i_iter - 100}-{round(self.scores_window.mean(), 2)}')
                break
