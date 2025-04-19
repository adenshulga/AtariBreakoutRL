import numpy as np


def merge_frame_stack_to_plot(frame_stack_obs: np.ndarray) -> float:
    """
    A helper function to plot a frame stack as a single human-interpretable image.

    Brighter pixels are more recent, pale pixels are older.
    Motions goes from pale to bright.

    Note! This function is designed for human vision convenience and it is NOT supposed to be used as part of
    data preprocessing for the Reinforcement Learning agent.
    """
    weights = np.ones(frame_stack_obs.shape[0], dtype=float)
    weights[-1] += weights.sum()
    weights /= weights.sum()
    result = (weights[:, None, None] * frame_stack_obs).sum(0)
    return result
