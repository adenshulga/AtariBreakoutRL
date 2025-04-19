from typing import Protocol

import gymnasium as gym
import numpy as np
import torch

from src.agent import DQNAgent


class ActionSampler(Protocol):
    """
    A protocol which defines an Callable which samples actions from states
    """

    def __call__(self, state: gym.wrappers.FrameStackObservation) -> int: ...


class RandomActionSampler:
    """
    We will need this guy to fill the buffer with initial 50-200K observations from a random policy.
    """

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def __call__(self, state: gym.wrappers.FrameStackObservation) -> int:
        action = self.action_space.sample()
        return action


class DqnActionSampler:
    """
    DQNAgent works on batched np.ndarray inputs.
    This class uses a DQNAgent to sample actions from single LazyFrames observations.

    This will be an epsilon-greedy sampler.
    A greedy sampler can be defined as well, but we won't need it.
    """

    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def __call__(self, state: gym.wrappers.FrameStackObservation) -> int:
        state_batched = np.array(state)[None]
        action_batched = self.agent.sample_actions(state_batched, greedy=False)
        action = action_batched.item()
        return int(action)
