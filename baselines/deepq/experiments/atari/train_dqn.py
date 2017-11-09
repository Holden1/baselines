import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
from datetime import datetime
import sys
import time
import json
import cv2
import threading
import copy

import baselines.common.tf_util as U

from baselines import logger, logger_utils
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.common.azure_utils import Container
from baselines.deepq.experiments.atari.model import model, dueling_model
from gameState import dsgym

ACTIONS = 9  # Now also action for doing nothing
MAX_NUM_TRAIN_ITERATIONS=100

def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Darksouls", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(300000), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(2e8),
                        help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=1,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=40000,
                        help="number of iterations between every target network update")
    parser.add_argument("--param-noise-update-freq", type=int, default=50,
                        help="number of iterations between every re-scaling of the parameter noise")
    parser.add_argument("--param-noise-reset-freq", type=int, default=10000,
                        help="maximum number of steps to take per episode before re-perturbing the exploration policy")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=True, help="whether or not to use dueling model")
    boolean_flag(parser, "prioritized", default=True, help="whether or not to use prioritized replay buffer")
    boolean_flag(parser, "param-noise", default=False,
                 help="whether or not to use parameter space noise for exploration")
    boolean_flag(parser, "layer-norm", default=False,
                 help="whether or not to use layer norm (should be True if param_noise is used)")
    boolean_flag(parser, "gym-monitor", default=False,
                 help="whether or not to use a OpenAI Gym monitor (results in slower training due to video recording)")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6,
                        help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4,
                        help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6,
                        help="eps parameter for prioritized replay buffer")
    # Checkpointing
    boolean_flag(parser, "dont-load-replay", default=False, help="whether to load replay buffer")
    boolean_flag(parser, "dont-load-num-iters", default=False, help="whether to load num iters")
    parser.add_argument("--overwrite-load-dir", type=str, default=None,
                        help="directory to force load model from and continue training")
    parser.add_argument("--save-dir", type=str, default="D:\openAi",
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--load-model-only-dir", type=str, default=None,
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e5,
                        help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True,
                 help="if true and model was previously saved then training will be resumed")
    boolean_flag(parser, "test", default=False,
                 help="fast testing, dont load prev model")
    return parser.parse_args()


# def make_env(game_name):
#     env = gym.make(game_name + "NoFrameskip-v4")
#     monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
#     env = wrap_dqn(monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
#     return env, monitored_env


def maybe_save_model(savedir, container, state):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    # Pause game while saving
    gymenv.pause_wrapper()
    model_dir = "model-{}".format(state["num_iters"])
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    if container is not None:
        container.put(os.path.join(savedir, model_dir), model_dir)
    relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
    if container is not None:
        container.put(os.path.join(savedir, 'training_state.pkl.zip'), 'training_state.pkl.zip')
    # relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'monitor_state.pkl'), 'monitor_state.pkl')
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))
    # unpause game
    gymenv.normal_speed_wrapper()


def maybe_load_model(savedir, container, onlymodeldir):
    """Load model if present at the specified path."""

    if savedir is None:
        return
    # elif onlymodeldir is not None:
    #    U.load_state(os.path.join("C:\openAi", "model-250000", "saved"))
    #    logger.log("Loaded models checkpoint at oops iterations")
    #    return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    if container is not None:
        logger.log("Attempting to download model from Azure")
        found_model = container.get(savedir, 'training_state.pkl.zip')
    else:
        found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"])
        if container is not None:
            container.get(savedir, model_dir)
        if onlymodeldir is None:
            U.load_state(os.path.join(savedir, model_dir, "saved"))
        else:
            print("Omlymodeldir:", onlymodeldir)
            U.load_state(os.path.join("D:\openAi", "model-" + onlymodeldir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state


def maybe_train(num_train):
    if (len(replay_buffer) > max(5 * args.batch_size, args.replay_buffer_size // 20)
        # and num_iters % args.learning_freq == 0
        ):
        # print("Training")
        # Sample a bunch of transitions from replay buffer
        if args.prioritized:
            experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
            (obses_t, feats_t, actions, rewards, obses_tp1, feats_tpl, dones, weights, batch_idxes) = experience
        else:
            obses_t, feats_t, actions, rewards, obses_tp1, feats_tpl, dones = replay_buffer.sample(args.batch_size)
            weights = np.ones_like(rewards)

        # Show images
        # cv2.imshow("baaa",obses_t[0][0])
        # cv2.waitKey(0)
        # cv2.imshow("baaa", obses_t[0][1])
        # cv2.waitKey(0)
        # cv2.imshow("baaa", obses_t[0][2])
        # cv2.waitKey(0)
        # cv2.imshow("baaa", obses_t[0][3])
        # cv2.waitKey(0)

        # Minimize the error in Bellman's equation and compute TD-error
        td_errors = train(obses_t, feats_t, actions, rewards, obses_tp1, feats_tpl, dones, weights)
        # Update the priorities in the replay buffer
        if args.prioritized:
            new_priorities = np.abs(td_errors) + args.prioritized_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)
        if num_train %50==0:
            print("Trained",num_train,"iterations...")
        return True
    return False


def maybe_train_for_seconds(num_seconds,num_train):
    t1 = datetime.now()
    if not maybe_train(num_train):
        print("sleeping for", num_seconds - (datetime.now() - t1).seconds, "seconds")
        time.sleep(num_seconds)
    else:
        for i in range(100):
            if i % 50 == 0:
                print("Training", 200 - i, "iterations")
            maybe_train(num_train)
        print("Done training...")


def train_while_resetting(num_train):
    t = threading.Thread(target=gymenv.reset, args=())
    t.daemon = True  # thread dies when main thread (only non-daemon thread) exits.
    t.start()
    #num_train=0
    # print("Is alive:",t.isAlive())
    while t.isAlive():
        if num_train<MAX_NUM_TRAIN_ITERATIONS and maybe_train(num_train):
            num_train+=1
        else:
            print("Done training",num_train,"iterations")
            break #We have trained all that we wanted, wait for reset thread to return
    t.join()


if __name__ == '__main__':
    args = parse_args()
    gymenv = dsgym()
    force_load_path = args.overwrite_load_dir
    if force_load_path is None:
        # Initialize logger
        logger.reset()
        logger_path = logger_utils.path_with_date(args.save_dir, args.env)
        logger.configure(logger_path, ["tensorboard", "stdout"])
        logger_utils.log_call_parameters(logger_path, args)
        savedir = logger_path
    else:
        print("FORCE load dir and continue training at {}".format(force_load_path))
        logger.configure(force_load_path, ["tensorboard", "stdout"])
        savedir = force_load_path

    # Parse savedir and azure container.
    savedir = args.save_dir
    if savedir is None:
        savedir = os.getenv('OPENAI_LOGDIR', None)
    if args.save_azure_container is not None:
        account_name, account_key, container_name = args.save_azure_container.split(":")
        container = Container(account_name=account_name,
                              account_key=account_key,
                              container_name=container_name,
                              maybe_create=True)
        if savedir is None:
            # Careful! This will not get cleaned up. Docker spoils the developers.
            savedir = tempfile.TemporaryDirectory().name
    else:
        container = None
    print("Savedir:", savedir)
    # Create and seed the env.
    # env, monitored_env = make_env(args.env)
    if args.seed > 0:
        set_global_seeds(args.seed)
        # env.unwrapped.seed(args.seed)

    # if args.gym_monitor and savedir:
    #     env = gym.wrappers.Monitor(env, os.path.join(savedir, 'gym_monitor'), force=True)

    if savedir:
        with open(os.path.join(savedir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    with U.make_session(4) as sess:
        # Create training graph and replay buffer
        def model_wrapper(img_in, features, num_actions, scope, **kwargs):
            actual_model = dueling_model if args.dueling else model
            return actual_model(img_in, features, num_actions, scope, layer_norm=args.layer_norm, **kwargs)


        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: U.Uint8Input((119, 70, 4), name=name),
            make_feature_ph=lambda name: U.Uint8Input((9 * 4,), name=name),
            q_func=model_wrapper,
            num_actions=ACTIONS,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=args.double_q,
            param_noise=args.param_noise
        )

        approximate_num_iters = args.num_steps / 4
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (approximate_num_iters / 50, 0.1),
            (approximate_num_iters / 5, 0.05)
        ], outside_value=0.01)  # final eps

        if args.prioritized:
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(args.replay_buffer_size)

        U.initialize()
        update_target()
        num_iters = 0

        # Load the model
        if not args.test:
            state = maybe_load_model(savedir, container, args.load_model_only_dir)
            if state is not None:
                if not args.dont_load_num_iters:
                    num_iters = state["num_iters"]
                    print("Loaded num iters")
                else:
                    print("Did not load num iters!")
                if not args.dont_load_replay:
                    replay_buffer = state["replay_buffer"]
                    print("Loaded replay buffer")

                else:
                    print("Did not load replay buffer!")
                    # monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = None, None
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs, feats, rew, done = gymenv.frame_step(6)
        # print("Done------------------- : ",done)
        if done:
            gymenv.reset()
            obs, feats, rew, done = gymenv.frame_step(6)
        num_iters_since_reset = 0
        reset = True
        info = {"steps": 0, "rewards": []}

        reward_since_reset = 0

        # Main trianing loop
        while True:
            num_iters += 1
            num_iters_since_reset += 1
            info['steps'] = num_iters
            # print(num_iters)

            # Take action and store transition in the replay buffer.
            kwargs = {}
            if not args.param_noise:
                update_eps = exploration.value(num_iters)
                update_param_noise_threshold = 0.
            else:
                if args.param_noise_reset_freq > 0 and num_iters_since_reset > args.param_noise_reset_freq:
                    # Reset param noise policy since we have exceeded the maximum number of steps without a reset.
                    reset = True

                update_eps = 0.01  # ensures that we cannot get stuck completely
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(
                    1. - exploration.value(num_iters) + exploration.value(num_iters) / float(ACTIONS))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = (num_iters % args.param_noise_update_freq == 0)

            action = act(np.array(obs)[None], np.array(feats)[None], update_eps=update_eps, **kwargs)[0]
            reset = False
            new_obs, new_feats, rew, done = gymenv.frame_step(action)
            reward_since_reset += rew
            replay_buffer.add(copy.copy(obs), copy.copy(feats), action, rew, copy.copy(new_obs), copy.copy(new_feats),
                              float(done))
            obs = new_obs
            if done:
                num_iters_since_reset = 0
                info["rewards"].append(reward_since_reset)
                reward_since_reset = 0
                start_train = time.time()
                may_kill_process = True
                num_train=0
                while not gymenv.can_reset():
                    if (time.time() - start_train) > 30 and may_kill_process:
                        # Assume loading screen is stuck
                        print("loading screen stuck, killing processes:", may_kill_process)
                        gymenv.kill_processes()
                        may_kill_process = False  # only kill process' one time
                    elif num_train<MAX_NUM_TRAIN_ITERATIONS: #We do not wish to exceed 250 training iterations (keep below 8 times per sample)
                        maybe_train(num_train)
                        num_train+=1
                    else:
                        time.sleep(1)
                #print("Trained",num_train,"iterations before reset")
                # Train for another x seconds to make sure loading screen is over
                #maybe_train_for_seconds(num_seconds=10,num_train)
                train_while_resetting(num_train)

                obs, feats, _, _ = gymenv.frame_step(6)
                reset = True

            # maybe_train() #- now only training while loading screen

            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if start_time is not None:
                steps_per_iter.update(info['steps'] - start_steps)
                iteration_time_est.update(time.time() - start_time)
            start_time, start_steps = time.time(), info["steps"]

            # Save the model and training state.
            if num_iters > 0 and (num_iters % args.save_freq == 0 or info["steps"] > args.num_steps):
                maybe_save_model(savedir, container, {
                    'replay_buffer': replay_buffer,
                    'num_iters': num_iters,
                    # 'monitor_state': monitored_env.get_state(),
                })

            if info["steps"] > args.num_steps:
                break

            if done:
                # print("Inside done")
                steps_left = args.num_steps - info["steps"]
                completion = np.round(info["steps"] / args.num_steps, 1)
                q_vals = debug["q_values"](np.asarray(obs).reshape(-1, 119, 70, 4), np.asarray(feats).reshape(-1, 36))

                logger.record_tabular("max q", np.max(q_vals))
                logger.record_tabular("avg q", np.mean(q_vals))

                logger.record_tabular("% completion", completion)
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))
                logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-100:]))
                logger.record_tabular("reward ", info["rewards"][-1])
                logger.record_tabular("exploration", exploration.value(num_iters))
                if args.prioritized:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else sys.maxsize)

                # Check if we killed boss, equal to reward in last step
                logger.record_tabular("Killed boss", rew)

                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()
