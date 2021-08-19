import json
import torch

from lib.helper import parse_config_for, extract_config_from
from lib.RLAgentTrainer import RLAgentTrainer
from lib.env.ParallelAgentsUnityEnvironment import ParallelAgentsUnityEnvironment
from lib.policy.ContinuousDiagonalGaussianPolicy import ContinuousDiagonalGaussianPolicy
from lib.function.ValueFunction import ValueFunction
from lib.ppo.PPO_ActorCriticAgent import PPO_ActorCriticRLAgent
from lib.log.WandbLogger import WandbLogger

if __name__ == "__main__":
    args = parse_config_for(
        program_name='Reacher PPO RL agent trainer',
        config_objects={
                "n_iterations": 10000,
                "max_t": 100,
                "enable_log": 0,
                "discount_rate": 0.99,
                "api_key": ""
            })

    env = ParallelAgentsUnityEnvironment(env_binary_path='Reacher_Linux_NoVis/Reacher.x86_64')
    get_policy = lambda: ContinuousDiagonalGaussianPolicy(state_size=env.state_size, action_size=env.action_size, seed=42,
                                              output_transform=lambda x: torch.tanh(x))
    get_value_function = lambda: ValueFunction(state_size=env.state_size, seed=42)
    agent = PPO_ActorCriticRLAgent(get_actor=get_policy, get_critic=get_value_function, learning_rate=0.01)

    config = extract_config_from(env, get_policy(), get_value_function(),
                                 agent, {"n_iterations": args.n_iterations, "max_t": args.max_t})

    print(f"initialized agent with config: \n {json.dumps(config, sort_keys=True, indent=4)}")

    logger = WandbLogger(
        wandb_project_name="udacity-drlnd-p2-reacher-ppo",
        api_key=args.api_key,
        config=config) if bool(args.enable_log) else None

    trainer = RLAgentTrainer(agent=agent, env=env, logger=logger)
    trainer.train(n_iterations=args.n_iterations, max_t=args.max_t)

    env.dispose()
    logger.dispose()
