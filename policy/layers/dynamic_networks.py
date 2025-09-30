import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inverse, matmul, transpose
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base
from policy.layers.building_blocks import MLP


class DynamicLearner(Base):
    def __init__(
        self,
        x_dim: int,
        action_dim: int,
        hidden_dim: list,
        nupdates: int,
        drop_out: float | None = None,
        activation: nn.Module = nn.ReLU(),
        Dynamic_lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
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
        self.nupdates = nupdates

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

        self.Dynamic_optimizer = torch.optim.Adam(
            params=self.parameters(), lr=Dynamic_lr
        )
        self.lr_scheduler = LambdaLR(self.Dynamic_optimizer, lr_lambda=self.lr_lambda)

        self.name = "DynamicLearner"
        self.device = device
        self.to(self.device)

    def lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def to_tensor(self, data):
        return torch.from_numpy(data).to(self._dtype).to(self.device)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the dynamics learner.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, x_dim).

        Returns:
            f (torch.Tensor): Bias term of shape (batch_size, x_dim).
            B (torch.Tensor): Action-dependent transformation matrix of shape (batch_size, x_dim, action_dim).
        """

        if not isinstance(x, torch.Tensor):
            x = self.to_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        n = x.shape[0]

        f = self.f(x).reshape(n, self.x_dim)
        B = self.B(x).reshape(n, self.x_dim, self.action_dim)
        Bbot = self.compute_B_perp_batch(B, self.x_dim - self.action_dim)

        return f, B, Bbot

    def learn(self, batch: dict) -> dict:
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        x = self.to_tensor(batch["x"])
        u = self.to_tensor(batch["u"])
        x_dot = self.to_tensor(batch["x_dot"])

        n = x.shape[0]

        f_approx = self.f(x)  # Compute bias term
        B_approx = self.B(x).reshape(
            n, self.x_dim, self.action_dim
        )  # Reshape output into dynamics matrix

        x_dot_approx = f_approx + matmul(B_approx, u.unsqueeze(-1)).squeeze(-1)

        loss = F.mse_loss(x_dot, x_dot_approx)

        self.Dynamic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.Dynamic_optimizer.step()
        self.lr_scheduler.step()

        loss_dict = {
            f"{self.name}/Dynamic_loss/loss": loss.item(),
            f"{self.name}/learning_rate/D_lr": self.Dynamic_optimizer.param_groups[0][
                "lr"
            ],
        }

        # Cleanup
        self.eval()
        update_time = time.time() - t0

        return loss_dict, update_time
