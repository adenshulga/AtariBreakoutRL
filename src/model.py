import torch
import torch.nn as nn

from config.main_cfg import ConvBackboneConfig, DuelingDqnConfig, ModelConfig
from config.constants import MAX_UINT_8


class ConvBackbone(nn.Sequential):
    """
    The convolutional part of a DQN model.
    Please, don't think about input scaling here: it will be implemented below.
    """

    def __init__(self, cfg: ConvBackboneConfig) -> None:
        super().__init__(
            nn.Conv2d(cfg.in_channels, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )


class DuelingDqnHead(nn.Module):
    """
    Implenets the Dueling DQN logic.
    Please, don't think about gradient scaling here (if you know what it is about): it will be implemented below.
    """

    def __init__(self, n_actions, cfg: DuelingDqnConfig) -> None:
        super().__init__()
        self.adv_stream = nn.Sequential(
            nn.Linear(in_features=cfg.inp_size, out_features=cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=cfg.hidden_size, out_features=n_actions),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(in_features=cfg.inp_size, out_features=cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=cfg.hidden_size, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, x.shape  # (batch_size, n_features)
        # your code
        # When calculating the mean advantage, please, remember, x is a batched input!
        value_func = self.value_stream(x)
        adv_func = self.adv_stream(x)

        element_wise_mean = adv_func.mean(dim=1, keepdim=True)

        adv_func_adjusted = adv_func - element_wise_mean

        return value_func + adv_func_adjusted


class InputScaler(nn.Module):
    def __init__(self, mult=1 / MAX_UINT_8):
        super().__init__()
        self.mult = mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mult


class GradScalerFunctional(torch.autograd.Function):
    """
    A torch.autograd.Function works as Identity on forward pass
    and scales the gradient by scale_factor on backward pass.
    """

    @staticmethod
    def forward(ctx, input, scale_factor):
        ctx.scale_factor = scale_factor
        return input

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        scale_factor = ctx.scale_factor
        grad_input = grad_output * scale_factor
        return grad_input, None


class GradScaler(nn.Module):
    """
    An nn.Module incapsulating GradScalerFunctional
    """

    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return GradScalerFunctional.apply(x, self.scale_factor)


class DQNetworkDueling(nn.Sequential):
    def __init__(self, n_actions: int, cfg: ModelConfig) -> None:
        input_scaler = InputScaler()  # the inputs come from the uint8 range
        backbone = ConvBackbone(cfg.backbone)  # your code
        grad_scaler = GradScaler(
            cfg.grad_scaler
        )  # Dueling DQN suggests do scale the gradient by 1 / sqrt(2)
        head = DuelingDqnHead(n_actions, cfg.queling_dqn)
        super().__init__(input_scaler, backbone, grad_scaler, head)
