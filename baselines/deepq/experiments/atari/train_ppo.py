#!/usr/bin/env python
import sys

import time
from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
import gym, logging
#from baselines import logger
from gameState import dsgym

def wrap_train(env):
    from baselines.common.atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, clip_rewards=True)
    env = FrameStack(env, 4)
    return env

def train(env_id, num_frames, seed):
    from baselines.ppo1 import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank != 0: logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
    gym.logger.setLevel(logging.WARN)
    num_timesteps = int(num_frames / 4 * 1.1)
    try:
        pposgd_simple.learn(dsgym(), policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=256,
            clip_param=0.2, entcoeff=0.01,
            optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
            gamma=0.99, lam=0.95,
            schedule='linear'
        )
    except KeyboardInterrupt:
        print("Saving on keyinterrupt")
        U.save_state("D:/openAi/ppo/" +str(time.time())+ "/saved_model")
        # quit
        sys.exit()
    U.save_state("D:/openAi/ppo/" +str(time.time()) + "/saved_model")

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Darksouls')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train(args.env, num_frames=40e6, seed=args.seed)

if __name__ == '__main__':
    main()
