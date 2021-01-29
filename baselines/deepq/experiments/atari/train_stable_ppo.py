#!/usr/bin/env python
import sys

import time
from gameState import dsgym
from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger

def wrap_train(env):
    from baselines.common.atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, clip_rewards=True)
    env = FrameStack(env, 4)
    return env

def train(num_timesteps):
    
    try:
        model = PPO1(MlpPolicy, dsgym(), timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
                 optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear')
        model.learn(total_timesteps=num_timesteps)
    except KeyboardInterrupt:
        print("Saving on keyinterrupt")
        model.save("ppo_darksouls_stable")
        # quit
        sys.exit()
    model.save("ppo_darksouls_stable")

def main():
    logger.configure()
    train(num_timesteps=40e6)

if __name__ == '__main__':
    main()
