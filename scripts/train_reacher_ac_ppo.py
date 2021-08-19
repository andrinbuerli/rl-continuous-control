import json
import torch
import numpy as np
import sys

sys.path.append("../")

from lib.helper import parse_config_for, extract_config_from
from lib.RLAgentTrainer import RLAgentTrainer
from lib.env.ParallelAgentsUnityEnvironment import ParallelAgentsUnityEnvironment
from lib.policy.ContinuousDiagonalGaussianPolicy import ContinuousDiagonalGaussianPolicy
from lib.function.ValueFunction import ValueFunction
from lib.ppo.PPO_ActorCriticAgent import PPO_ActorCriticRLAgent
from lib.log.WandbLogger import WandbLogger

if __name__ == "__main__":
    print(f"Found {torch._C._cuda_getDeviceCount()} GPU")

    args = parse_config_for(
        program_name='Reacher PPO Actor Critic style RL agent trainer',
        config_objects={
            "discount_rate": 0.99,
            "epsilon": 0.01,
            "epsilon_decay": 0.99,
            "beta": 0.1,
            "beta_deay": 0.995,
            "learning_rate": 0.001,
            "SGD_epoch": 4,
            "n_iterations": 100000,
            "max_t": 500,
            "gae_lambda": 0.9,
            "enable_log": 1,
            "api_key": "",
            "seed": int(np.random.randint(0, 1e10, 1)[0])
        })

    env = ParallelAgentsUnityEnvironment(target_reward=35,
                                         env_binary_path='../environments/Reacher_Linux_NoVis/Reacher.x86_64')

    policy = ContinuousDiagonalGaussianPolicy(state_size=env.state_size, action_size=env.action_size,
                                              seed=args.seed, output_transform=lambda x: torch.tanh(x))
    value_function = ValueFunction(state_size=env.state_size, seed=args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent = PPO_ActorCriticRLAgent(
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
        device=device)

    if device != "cpu":
        torch.cuda.set_device(0)

    config = extract_config_from(env, policy, value_function,
                                 agent, {"n_iterations": args.n_iterations,
                                         "max_t": args.max_t,
                                         "seed": args.seed
                                         })

    print(f"initialized agent with config: \n {json.dumps(config, sort_keys=True, indent=4)}")

    logger = WandbLogger(
        wandb_project_name="udacity-drlnd-p2-reacher-ppo",
        run_name="PPO A2C",
        api_key=args.api_key,
        config=config) if bool(args.enable_log) else None

    trainer = RLAgentTrainer(agent=agent, env=env, logger=logger)
    trainer.train(n_iterations=args.n_iterations, max_t=args.max_t)

    env.dispose()
    logger.dispose()