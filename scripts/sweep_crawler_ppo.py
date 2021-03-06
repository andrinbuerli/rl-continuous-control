import json
import torch
import numpy as np
import wandb
import sys

sys.path.append("../")

from lib.helper import parse_config_for, extract_config_from
from lib.RLAgentTrainer import RLAgentTrainer
from lib.env.ParallelAgentsUnityEnvironment import ParallelAgentsUnityEnvironment
from lib.models.policy.StochasticContinuousGaussianPolicy import StochasticContinuousGaussianPolicy
from lib.models.function import StateValueFunction
from lib.agent.ppo.PPOActorCriticRLAgent import PPOActorCriticRLAgent
from lib.log.WandbLogger import WandbSweepLogger

if __name__ == "__main__":
    print(f"Found {torch._C._cuda_getDeviceCount()} GPU")

    args = parse_config_for(
        program_name='Reacher PPO Actor Critic style RL agent trainer',
        config_objects={
            "discount_rate": 0.99,
            "epsilon": 0.2,
            "epsilon_decay": 0.9995,
            "beta": .2,
            "beta_deay": 0.9995,
            "learning_rate": 0.0005,
            "SGD_epoch": 4,
            "n_iterations": 1000000,
            "max_t": 2048,
            "gae_lambda": 0.9,
            "enable_log": 1,
            "critic_loss_coefficient": .5,
            "api_key": "",
            "seed": int(np.random.randint(0, 1e10, 1)[0])
        })

    # Pass them to wandb.init
    wandb.init(config=args, entity="andrinburli")
    # Access all hyperparameter values through wandb.config
    config = wandb.config

    env = ParallelAgentsUnityEnvironment(
        name="Crawler",
        target_reward=3000,
        env_binary_path='environments/Crawler_Linux_NoVis/Crawler.x86_64')

    policy = StochasticContinuousGaussianPolicy(state_size=env.state_size, action_size=env.action_size,
                                                seed=args.seed, output_transform=lambda x: torch.tanh(x))
    value_function = StateValueFunction(state_size=env.state_size, seed=args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent = PPOActorCriticRLAgent(
        actor=policy,
        critic=value_function,
        discount_rate=args.discount_rate,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        beta=args.beta,
        beta_decay=args.beta_deay,
        learning_rate=args.learning_rate,
        SGD_epoch=args.SGD_epoch,
        gae_lambda=args.gae_lambda,
        critic_loss_coefficient=args.critic_loss_coefficient,
        device=device)

    if device != "cpu":
        torch.cuda.set_device(0)

    config = extract_config_from(env, policy, value_function,
                                 agent, {"n_iterations": args.n_iterations,
                                         "max_t": args.max_t,
                                         "seed": args.seed
                                         })

    print(f"initialized agent with config: \n {json.dumps(dict(config), sort_keys=True, indent=4)}")

    logger = WandbSweepLogger(config=config)

    trainer = RLAgentTrainer(agent=agent, env=env, logger=logger, seed=args.seed)
    trainer.train(n_iterations=args.n_iterations, max_t=args.max_t)

    env.dispose()
    logger.dispose()
