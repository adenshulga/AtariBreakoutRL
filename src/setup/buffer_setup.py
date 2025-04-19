from config.main_cfg import ReplayBufferConfig
from src.dqn.replay_buffer import ReplayBuffer
from src.dqn.utils import is_enough_ram
from tqdm import trange
from src.evaluation import play_and_record
import gymnasium as gym
from src.action_sampler import ActionSampler


def setup_buffer(
    env: gym.Env,
    action_sampler: ActionSampler,
    cfg: ReplayBufferConfig,
    min_available_gb: float,
    seed: int,
) -> ReplayBuffer:
    state, _ = env.reset(seed=seed)

    exp_replay = ReplayBuffer(cfg.size)
    for i in trange(cfg.initial_fill // cfg.initial_game_duration):
        if not is_enough_ram(min_available_gb=min_available_gb):
            print("""
                Less than 100 Mb RAM available.
                Make sure the buffer size in not too huge.
                Also check, maybe other processes consume RAM heavily.
                """)
            break
        play_and_record(
            state,
            action_sampler,
            env,
            exp_replay,
            n_steps=cfg.initial_game_duration,
        )
        if len(exp_replay) >= cfg.initial_fill:
            break

    return exp_replay
