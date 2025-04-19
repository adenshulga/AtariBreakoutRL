from config.main_cfg import AgentConfig, ModelConfig
from src.agent import DQNAgent
from src.model import DQNetworkDueling
import typing as tp

MainNetwork: tp.TypeAlias = DQNetworkDueling
TargetNetwork: tp.TypeAlias = DQNetworkDueling


def setup_networks(n_actions, cfg: ModelConfig) -> tuple[MainNetwork, TargetNetwork]:
    q_network = DQNetworkDueling(n_actions, cfg)

    target_network = DQNetworkDueling(n_actions, cfg)
    target_network.load_state_dict(q_network.state_dict())

    return q_network, target_network


def setup_agent(q_network: DQNetworkDueling, cfg: AgentConfig) -> DQNAgent:
    return DQNAgent(q_network, cfg)


# def setup_sampler()
