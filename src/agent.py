import torch.nn as nn
import numpy as np
import torch
from config.main_cfg import AgentConfig


class DQNAgent(nn.Module):
    """
    Epsilon-greedy policy with a torch.nn.Module Q-value estimator.
    """

    def __init__(self, q_network: nn.Module, cfg: AgentConfig) -> None:
        super().__init__()
        self.epsilon = cfg.epsilon
        self.q_network = q_network

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        qvalues = self.q_network(state_t)
        return qvalues

    @torch.no_grad()  # we don't need autograd here, so let's save the computations
    def get_qvalues(self, states: np.ndarray) -> np.ndarray:
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states_pt = torch.tensor(
            np.array(states), device=model_device, dtype=torch.float32
        )
        # Use your network to compute qvalues for given state
        qvalues_pt = self(states_pt)
        qvalues = qvalues_pt.data.cpu().numpy()
        return qvalues

    def sample_actions_by_qvalues(
        self, qvalues: np.ndarray, greedy: bool = False
    ) -> np.ndarray:
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy."""
        batch_size, n_actions = qvalues.shape
        greedy_actions = qvalues.argmax(axis=-1)  # your code
        if greedy:
            return greedy_actions

        random_actions = np.random.randint(low=0, high=n_actions, size=batch_size)
        should_explore = np.random.binomial(1, self.epsilon, size=batch_size).astype(
            bool
        )
        epsilon_greedy_actions = np.where(
            should_explore, random_actions, greedy_actions
        )
        return epsilon_greedy_actions

    def sample_actions(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        qvalues = self.get_qvalues(states)
        actions = self.sample_actions_by_qvalues(qvalues, greedy)
        return actions
