import json
import wandb
import torch
import numpy as np
import sys

sys.path.append("../")

from lib.helper import parse_config_for
from lib.RLAgentTrainer import RLAgentTrainer
from lib.env.ParallelAgentsUnityEnvironment import ParallelAgentsUnityEnvironment
from lib.policy.ContinuousDiagonalGaussianPolicy import ContinuousDiagonalGaussianPolicy
from lib.ppo.PPORLAgent import PPORLAgent
from lib.log.WandbLogger import WandbSweepLogger

if __name__ == "__main__":
    print(f"Found {torch._C._cuda_getDeviceCount()} GPUS")
    args = parse_config_for(
        program_name='Reacher PPO RL agent trainer',
        config_objects={
            "discount_rate": 0.1,
            "epsilon": 0.1,
            "epsilon_decay": 0.1,
            "beta": 0.1,
            "beta_deay": 0.1,
            "learning_rate": 0.1,
            "SGD_epoch": 1,
            "n_iterations": 1,
            "max_t": 1,
            "seed": int(np.random.randint(0, 1e10, 1)[0])
        })

    # Pass them to wandb.init
    wandb.init(config=args)
    # Access all hyperparameter values through wandb.config
    config = wandb.config

    env = ParallelAgentsUnityEnvironment(
        name="Reacher",
        target_reward=35,
        env_binary_path='../environments/Reacher_Linux_NoVis/Reacher.x86_64')
    policy = ContinuousDiagonalGaussianPolicy(state_size=env.state_size, action_size=env.action_size, seed=args.seed,
                                              output_transform=lambda x: torch.tanh(x))
    agent = PPORLAgent(
        policy=policy,
        discount_rate=args.discount_rate,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        beta=args.beta,
        beta_decay=args.beta_deay,
        learning_rate=args.learning_rate,
        SGD_epoch=args.SGD_epoch,
        device="cuda:0")
    torch.cuda.set_device(0)
    print(f"initialized agent with config: \n {json.dumps(dict(config), sort_keys=True, indent=4)}")

    logger = WandbSweepLogger(config=config)

    trainer = RLAgentTrainer(agent=agent, env=env, logger=logger, seed=args.seed)
    trainer.train(n_iterations=args.n_iterations, max_t=args.max_t)

    env.dispose()
