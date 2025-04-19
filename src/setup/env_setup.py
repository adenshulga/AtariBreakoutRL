import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
from src.dqn.atari_wrappers import FireResetEnv
from src.dqn.atari_wrappers import EpisodicLifeEnv
from config.main_cfg import EnvConfig

gym.register_envs(ale_py)


def make_basic_env(cfg: EnvConfig):
    return gym.make(cfg.name, render_mode="rgb_array")


def apply_gray_scale_wrap(env):
    # With the argument values chosen as below, the gym.wrappers.AtariPreprocessing wrapper
    # only converts images to grayscale and downsamples them the screen_size
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=0,  # the default value 30 can be harmful with FireResetEnv and frame_skip=5
        frame_skip=1,  # frame_skip has already been set to 5 inside the env
        terminal_on_life_loss=False,  # we do this explicitly in the FireResetEnv wrapper
        screen_size=64,  # please use 84 (which is the standard value) or 64 (which will save some computations and memory)
    )
    return env


def apply_atary_specific_wrap(env):
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    return env


def make_final_env(cfg: EnvConfig):
    """
    Builds the environment with all the wrappers applied.
    The environment is meant be used directly as an RL algorithm input.

    apply_frame_stack=False can be useful for vecotrized environments, which are not required for this assignment.
    """
    env = make_basic_env(cfg)
    env = apply_gray_scale_wrap(env)
    env = apply_atary_specific_wrap(env)
    if cfg.n_frames_stacked is not None:
        env = gym.wrappers.FrameStackObservation(env, stack_size=cfg.n_frames_stacked)
    return env
