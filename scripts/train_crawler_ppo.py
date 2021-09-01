import json
import torch
import numpy as np
import sys

sys.path.append("../")

from lib.helper import parse_config_for, extract_config_from
from lib.RLAgentTrainer import RLAgentTrainer
from lib.env.ParallelAgentsUnityEnvironment import ParallelAgentsUnityEnvironment
from lib.models.policy.StochasticContinuousGaussianPolicy import StochasticContinuousGaussianPolicy
from lib.models.function import StateValueFunction
from lib.models.PPOActorCriticJointModel import PPOActorCriticJointModel
from lib.agent.ppo.PPOActorCriticRLAgent import PPOActorCriticRLAgent
from lib.log.WandbLogger import WandbLogger

if __name__ == "__main__":
    print(f"Found {torch._C._cuda_getDeviceCount()} GPU")

    args = parse_config_for(
        program_name='Crawler PPO Actor Critic style RL agent trainer',
        config_objects={
            "discount_rate": 0.95,
            "epsilon": 0.1,
            "epsilon_decay": .995,
            "beta": 0.01,
            "beta_deay": .995,
            "learning_rate": 0.0001,
            "batch_size": 1024,
            "SGD_epoch": 32,
            "n_iterations": int(1e7),
            "max_t": 512,
            "gae_lambda": 0.99,
            "enable_log": 1,
            "critic_loss_coefficient": .5,
            "api_key": "",
            "seed": int(np.random.randint(0, 1e10, 1)[0])
        })

    env = ParallelAgentsUnityEnvironment(
        name="Crawler",
        target_reward=3000,
        env_binary_path='../environments/Crawler_Linux_NoVis/Crawler.x86_64')

    model = PPOActorCriticJointModel(state_size=env.state_size, action_size=env.action_size,
                                     seed=args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent = PPOActorCriticRLAgent(
        model=model,
        discount_rate=args.discount_rate,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        beta=args.beta,
        beta_decay=args.beta_deay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        SGD_epoch=args.SGD_epoch,
        gae_lambda=args.gae_lambda,
        critic_loss_coefficient=args.critic_loss_coefficient,
        device=device,
        grad_clip_max=1.0)

    if device != "cpu":
        torch.cuda.set_device(0)

    config = extract_config_from(env, model,
                                 agent, {"n_iterations": args.n_iterations,
                                         "max_t": args.max_t,
                                         "seed": args.seed
                                         })

    print(f"initialized agent with config: \n {json.dumps(config, sort_keys=True, indent=4)}")

    logger = WandbLogger(
        wandb_project_name="udacity-drlnd-p2-crawler-ppo-v3",
        run_name="PPO A2C",
        api_key=args.api_key,
        entity="andrinburli",
        config=config) if bool(args.enable_log) else None

    trainer = RLAgentTrainer(agent=agent, env=env, logger=logger, seed=args.seed)
    trainer.train(n_iterations=args.n_iterations, max_t=args.max_t)

    env.dispose()
    logger.dispose()
