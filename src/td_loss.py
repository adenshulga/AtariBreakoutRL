import torch
import torch.nn as nn


def compute_td_loss_on_tensors(
    states: torch.Tensor,  # (batch_size, *state_shape)
    actions: torch.Tensor,  # (batch_size,)
    rewards: torch.Tensor,  # (batch_size,)
    next_states: torch.Tensor,  # (batch_size, *state_shape)
    is_done: torch.Tensor,  # (batch_size,), torch.bool
    agent: nn.Module,
    target_network: nn.Module,
    gamma: float = 0.99,
    check_shapes=False,
):
    predicted_qvalues = agent(states)  # shape: [batch_size, n_actions]
    assert is_done.dtype is torch.bool

    # compute q-values for all actions in next states
    with torch.no_grad():
        predicted_next_qvalues_target = target_network(
            next_states
        )  # batch_size x n_actions

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[
        range(len(states)), actions
    ]  # shape: [batch_size]

    # compute V*(next_states) using predicted next q-values
    next_state_values, _ = predicted_next_qvalues_target.max(dim=1)

    if check_shapes:
        assert (
            next_state_values.dim() == 1
            and next_state_values.shape[0] == states.shape[0]
        ), "must predict one value per state"
        assert not next_state_values.requires_grad

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    target_qvalues_for_actions = rewards + gamma * (~is_done) * next_state_values

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions) ** 2)

    if check_shapes:
        assert predicted_next_qvalues_target.data.dim() == 2, (
            "make sure you predicted q-values for all actions in next state"
        )
        assert next_state_values.data.dim() == 1, (
            "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        )
        assert target_qvalues_for_actions.data.dim() == 1, (
            "there's something wrong with target q-values, they must be a vector"
        )

    return loss


def compute_td_loss_on_tensors_double(
    states: torch.Tensor,  # (batch_size, *state_shape)
    actions: torch.Tensor,  # (batch_size,)
    rewards: torch.Tensor,  # (batch_size,)
    next_states: torch.Tensor,  # (batch_size, *state_shape)
    is_done: torch.Tensor,  # (batch_size,), torch.bool
    agent: nn.Module,
    target_network: nn.Module,
    gamma: float = 0.99,
    check_shapes=False,
):
    predicted_qvalues = agent(states)  # shape: [batch_size, n_actions]
    assert is_done.dtype is torch.bool

    # compute q-values for all actions in next states
    with torch.no_grad():
        predicted_next_qvalues_target = target_network(
            next_states
        )  # shape: [batch_size, n_actions]

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[
        range(len(states)), actions
    ]  # shape: [batch_size]

    # compute V*(next_states) using predicted next q-values
    opt_actions = agent(next_states).argmax(dim=1)  # batch_size
    next_state_values = predicted_next_qvalues_target[range(len(states)), opt_actions]

    if check_shapes:
        assert (
            next_state_values.dim() == 1
            and next_state_values.shape[0] == states.shape[0]
        ), "must predict one value per state"
        assert not next_state_values.requires_grad

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    target_qvalues_for_actions = rewards + gamma * (~is_done) * next_state_values

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions) ** 2)

    if check_shapes:
        assert predicted_next_qvalues_target.data.dim() == 2, (
            "make sure you predicted q-values for all actions in next state"
        )
        assert next_state_values.data.dim() == 1, (
            "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        )
        assert target_qvalues_for_actions.data.dim() == 1, (
            "there's something wrong with target q-values, they must be a vector"
        )

    return loss


def compute_td_loss(
    states,
    actions,
    rewards,
    next_states,
    is_done,
    agent,
    target_network,
    gamma=0.99,
    check_shapes=False,
    device=None,
    tensor_loss_evaluator=compute_td_loss_on_tensors_double,
):
    """Compute td loss using torch operations only. Use the formulae above."""

    if device is None:
        device = next(agent.parameters()).device
    states = torch.tensor(
        states, device=device, dtype=torch.float32
    )  # shape: [batch_size, *state_shape]
    actions = torch.tensor(
        actions, device=device, dtype=torch.int64
    )  # shape: [batch_size]
    rewards = torch.tensor(
        rewards, device=device, dtype=torch.float32
    )  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done, device=device, dtype=torch.bool
    )  # shape: [batch_size]

    return tensor_loss_evaluator(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        is_done=is_done,
        agent=agent,
        target_network=target_network,
        gamma=gamma,
        check_shapes=check_shapes,
    )
