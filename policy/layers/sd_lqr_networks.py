import torch
import torch.nn as nn

from policy.layers.building_blocks import MLP


class SDCLearner(nn.Module):
    def __init__(
        self,
        x_dim: int,
        a_dim: int,
        hidden_dim: list,
        drop_out: float = 0.2,
        activation: nn.Module = nn.Tanh(),
    ):
        super(SDCLearner, self).__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.model = MLP(
            2 * x_dim,
            hidden_dim,
            (a_dim + 1) * x_dim**2,
            activation=activation,
            dropout_rate=drop_out,
        )

    def forward(self, x: torch.Tensor):
        logits = self.model(x)

        # Split the logits into two parts
        Af = logits[:, : self.x_dim**2].reshape(-1, self.x_dim, self.x_dim)
        Bf = logits[:, self.x_dim**2 :].reshape(-1, self.a_dim, self.x_dim, self.x_dim)

        return Af, Bf
