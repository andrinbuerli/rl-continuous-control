import json
import torch
import numpy as np
import sys

sys.path.append("../")

from lib.helper import parse_config_for, extract_config_from
from lib.RLAgentTrainer import RLAgentTrainer
from lib.env.ParallelAgentsUnityEnvironment import ParallelAgentsUnityEnvironment
from lib.policy.DeterministicContinuousGaussianPolicy import DeterministicContinuousGaussianPolicy
from lib.function.StateActionValueFunction import StateActionValueFunction
from lib.agent.ddpg.DDPGRLAgent import DDPGRLAgent
from lib.log.WandbLogger import WandbLogger

if __name__ == "__main__":
    print(f"Found {torch._C._cuda_getDeviceCount()} GPU")

    args = parse_config_for(
        program_name='Crawler PPO Actor Critic style RL agent trainer',
        config_objects={
            "gamma": 0.99,
            "epsilon": 1,
            "epsilon_decay": .999995,
            "epsilon_min": 0.01,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "tau": 1e-3,
            "update_every": 1,
            "learning_rate": 0.0005,
            "update_for": 5,
            "n_iterations": int(1e7),
            "max_t": 40,
            "max_t_iteration": [2000, 4000, 6000, 8000],
            "enable_log": 1,
            "api_key": "",
            "seed": int(np.random.randint(0, 1e10, 1)[0]),
            "agent_weights": "agents/Crawler-DDPG_60753-227872720-99.11"
        })

    env = ParallelAgentsUnityEnvironment(
        name="Crawler",
        target_reward=3000,
        env_binary_path='../environments/Crawler_Linux_NoVis/Crawler.x86_64')

    policy = lambda: DeterministicContinuousGaussianPolicy(
        state_size=env.state_size, action_size=env.action_size,
        seed=args.seed, output_transform=lambda x: torch.tanh(x), reduced_capacity=False)
    value_function = lambda: StateActionValueFunction(
        state_size=env.state_size, action_size=env.action_size,
        seed=args.seed, reduced_capacity=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent = DDPGRLAgent(
        get_actor=policy,
        get_critic=value_function,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        lr=args.learning_rate,
        update_every=args.update_every,
        update_for=args.update_for,
        prioritized_exp_replay=False,
        device=device,
        action_size=env.action_size,
        state_size=env.state_size)

    if args.agent_weights is not None:
        agent.load(args.agent_weights)
        print(f"loaded agent weights from : {args.agent_weights}")

    if device != "cpu":
        torch.cuda.set_device(0)

    config = extract_config_from(env, policy, value_function,
                                 agent, {"n_iterations": args.n_iterations,
                                         "max_t": args.max_t,
                                         "seed": args.seed
                                         })

    print(f"initialized agent with config: \n {json.dumps(config, sort_keys=True, indent=4)}")

    logger = WandbLogger(
        wandb_project_name="udacity-drlnd-p2-crawler-ddpg-v1",
        run_name=None,
        entity="andrinburli",
        api_key=args.api_key,
        config=config) if bool(args.enable_log) else None

    trainer = RLAgentTrainer(agent=agent, env=env, logger=logger, seed=args.seed)
    trainer.train(n_iterations=args.n_iterations, max_t=args.max_t, max_t_iteration=args.max_t_iteration,
                  intercept=True)

    env.dispose()
    logger.dispose()
