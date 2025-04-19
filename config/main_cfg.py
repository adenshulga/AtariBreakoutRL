from dataclasses import dataclass, field
import typing as tp


@dataclass
class EnvConfig:
    name: tp.Literal["ALE/Breakout-v5"] = "ALE/Breakout-v5"
    n_frames_stacked: int | None = 4


@dataclass
class ConvBackboneConfig:
    in_channels: int = 4


@dataclass
class DuelingDqnConfig:
    inp_size: int = 64 * 7 * 7
    hidden_size: int = 512


@dataclass
class ModelConfig:
    queling_dqn: DuelingDqnConfig = field(default_factory=DuelingDqnConfig)
    backbone: ConvBackboneConfig = field(default_factory=ConvBackboneConfig)
    grad_scaler: float = 1 / 2**0.5


@dataclass
class AgentConfig:
    epsilon: float = 0.5
    model: ModelConfig = field(default_factory=ModelConfig)
    # greedy: bool = False


@dataclass
class ReplayBufferConfig:
    size: int = 10**6
    initial_fill = 200_000
    initial_game_duration = 100


@dataclass
class AdamOptimizerConfig:
    lr: float = 6.25e-05
    eps: float = 1.4e-4


@dataclass
class MainConfig:
    update_frequency: int = 4
    batch_size: int = 32
    total_steps: int = 10 * 10**6
    decay_steps: int = 10**6

    init_epsilon: float = 1  # Nature DQN
    final_epsilon: float = 0.1  # Nature DQN

    loss_freq: int = 100
    refresh_target_network_freq: int = 10_000  # Nature DQN
    eval_freq: int = 10_000

    max_grad_norm: float = 10  # Dueling DQN

    n_lives: int = 5

    use_tensorboard: bool = True

    seed: int = 1
    device: tp.Literal["cuda", "cpu"] = "cuda"
    min_available: float = 0.1
