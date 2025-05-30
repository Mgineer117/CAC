import torch
import torch.nn as nn

from policy.layers.building_blocks import MLP


class DynamicLearner(nn.Module):
    def __init__(
        self,
        x_dim: int,
        action_dim: int,
        hidden_dim: list,
        drop_out: float | None = None,
        activation: nn.Module = nn.Tanh(),
    ):
        """
        Neural network module that models the dynamics of a system.

        Args:
            x_dim (int): Dimension of the state/input vector.
            action_dim (int): Number of discrete actions or dimensions of the action space.
            hidden_dim (list): List of hidden layer sizes for the MLPs.
            drop_out (float, optional): Dropout rate, if any. Default is None (no dropout).
            activation (nn.Module): Activation function to use (default: nn.Tanh()).
        """
        super(DynamicLearner, self).__init__()

        self.x_dim = x_dim
        self.action_dim = action_dim
        self.activation = activation

        # State-dependent bias term for dynamics
        self.f = MLP(
            x_dim, hidden_dim, x_dim, activation=self.activation, dropout_rate=drop_out
        )

        # Action-dependent dynamics coefficients
        self.B = MLP(
            x_dim,
            hidden_dim,
            x_dim * action_dim,
            activation=self.activation,
            dropout_rate=drop_out,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the dynamics learner.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, x_dim).

        Returns:
            f (torch.Tensor): Bias term of shape (batch_size, x_dim).
            B (torch.Tensor): Action-dependent transformation matrix of shape (batch_size, x_dim, action_dim).
        """
        n = x.shape[0]

        f = self.f(x)  # Compute bias term
        B = self.B(x).reshape(
            n, self.x_dim, self.action_dim
        )  # Reshape output into dynamics matrix

        return f, B
