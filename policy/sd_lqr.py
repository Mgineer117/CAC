import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import solve_continuous_are
from torch.linalg import solve
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base


class SD_LQR(Base):
    def __init__(
        self,
        x_dim: int,
        action_dim: int,
        get_f_and_B: nn.Module,
        SDC_func: nn.Module,
        Q_scaler: float = 1.0,
        R_scaler: float = 1.0,
        device: str = "cpu",
    ):
        super(SD_LQR, self).__init__()

        """
        Do not use Multiprocessor => use less batch
        """
        # constants
        self.name = "SD_LQR"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = action_dim

        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            # set to eval mode due to dropout
            self.get_f_and_B.eval()
        self.SDC_func = SDC_func.eval()

        self.Q_scaler = Q_scaler
        self.R_scaler = R_scaler

        #
        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # shape: (1, state_dim)

        # Decompose state
        x, xref, uref, t = self.trim_state(state)

        if not deterministic:
            u = uref + torch.randn_like(uref) * 0.1
        else:
            e = x - xref  # shape: (1, x_dim)
            sdc_input = torch.concatenate((x, e), dim=-1)

            _, B, _ = self.get_f_and_B(x)

            Af, Bf = self.SDC_func(sdc_input)
            Bf_u = (uref.view(1, self.action_dim, 1, 1) * Bf).sum(dim=1)

            A = Af + Bf_u
            A, B = A.squeeze(), B.squeeze()

            # Solve Riccati equation: A^T P + P A - P B R^-1 B^T P + Q = -Q
            Q = (self.Q_scaler + 1e-5) * torch.eye(
                self.x_dim, dtype=self._dtype, device=self.device
            )
            R = (self.R_scaler + 1e-5) * torch.eye(
                self.action_dim, dtype=self._dtype, device=self.device
            )

            # Use SciPy solver for CARE
            A_np = A.detach().cpu().numpy()
            B_np = B.detach().cpu().numpy()
            Q_np = Q.detach().cpu().numpy()
            R_np = R.detach().cpu().numpy()
            P_np = solve_continuous_are(A_np, B_np, Q_np, R_np)
            P = torch.from_numpy(P_np).to(A)

            # Compute feedback gain: K = R^-1 B^T P
            K = solve(R, B.T @ P)  # shape: (u_dim, x_dim)

            # Compute LQR control law: u = uref - K @ e
            u = uref - (K @ e.unsqueeze(-1)).squeeze(-1)

        # Return
        return u, {
            "probs": self.dummy,
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

    def learn(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        pass
