import numpy as np
import torch
import gymnasium as gym
from src.action_sampler import ActionSampler


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000, seed=None):
    """Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward."""
    rewards = []
    for _ in range(n_games):
        s, _ = env.reset(seed=seed)
        reward = 0
        for _ in range(t_max):
            action = agent.sample_actions(np.array(s)[None], greedy=greedy)[0]
            s, r, terminated, truncated, _ = env.step(action)
            reward += r
            if terminated or truncated:
                break

        rewards.append(reward)
    return np.mean(rewards)


@torch.no_grad()
def play_and_record(
    initial_state: gym.wrappers.FrameStackObservation,
    action_sampler: ActionSampler,
    env,
    exp_replay,
    n_steps=1,
):
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends due to termination or truncation, add record with done=terminated and reset the game.
    It is guaranteed that env has terminated=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        action = action_sampler(s)
        new_s, r, term, trunc, _ = env.step(action)
        exp_replay.add(s, action, r, new_s, term or trunc)
        if term or trunc:
            env.reset()
        s = new_s
        sum_rewards += r

    return sum_rewards, s
