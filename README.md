
# Continuous Control

### Introduction

In this project, an agent is trained to interact in two different environments whereas the action space is continuous in both of them. 

#### Reacher

The first environment is called *Reacher*, here the task is to control a double-jointed arm such that it can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. There are 20 identical agents, each with its own copy of the environment interacting independently and synchronous.

|                   Initial (random) policy:                   |                       Learned policy:                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="imgs\initial_reacher.gif" style="height:200px;" /> | <img src="imgs\trained_reacher.gif"  style="height:200px;" /> |

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. The environment is considered solved if a mean reward of 35 is reached over 100 consecutive episodes.

#### Crawler

The second environment is called *Crawler*.  In this continuous control environment, the goal is to teach a creature with four legs to walk forward without falling.  The reward function is based on the following criterions:

- Body velocity matches goal velocity. (normalized between (0,1))
- Head direction alignment with goal direction. (normalized between (0,1))

It is a product of all the rewards, this helps the agent try to maximize all rewards instead of the easiest rewards. There are 12 identical agents, each with its own copy of the environment interacting independently and synchronous.

|        Initial (random) policy:         |               Learned policy:               |
| :-------------------------------------: | :-----------------------------------------: |
| <img src="imgs\initial_crawler.gif"  /> | <img src="imgs\trained_crawler.gif"  /> |

The observation space consists of 172 variables corresponding to position, rotation, velocity, and angular velocities of each limb plus the acceleration and angular acceleration of the body. Each action is a vector with 20 numbers, corresponding to target rotations for joints. The environment is considered solved if a mean reward of 3000 is reached over 100 consecutive episodes.

### Getting started

1. Install [Docker](https://docs.docker.com/get-docker/)

2. If you want to log the training process, create a [wandb](https://wandb.ai/site) account

3. Replace the `<API-KEY>` tag in the `docker-compose.yml` file with your wandb api key

4. Start the training with 

   ``` bash
   # train a ppo agent in reacher environment
   docker-compose up train_reacher_ppo
   # train a ddpg agent in reacher environment
   docker-compose up train_reacher_ddpg
   # train a ppo agent in crawler environment
   docker-compose up train_crawler_ppo
   # train a ddpg agent in crawler environment
   docker-compose up train_crawler_ddpg
   ```

If you want to visually watch a trained agent interacting with the environment, you can download the required files at

**Reacher:**

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

**Crawler:**

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Once that the visual environments have been downloaded, you can follow the instructions in the notebooks `Watch_Reacher.ipynb` or `Wach_crawler.ipynb`.
