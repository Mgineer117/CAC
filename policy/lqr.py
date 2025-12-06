import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import solve_continuous_are
from torch import inverse, matmul, transpose
from torch.autograd import grad
from torch.linalg import solve
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base


class LQR(Base):
    def __init__(
        self,
        x_dim: int,
        action_dim: int,
        get_f_and_B: Callable,
        Q_scaler: float = 1.0,
        R_scaler: float = 0.03,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        super(LQR, self).__init__()

        """
        Do not use Multiprocessor => use less batch
        """
        # constants
        self.name = "LQR"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = action_dim

        self.Q_scaler = Q_scaler
        self.R_scaler = R_scaler
        self.get_f_and_B = get_f_and_B

        self.gamma = gamma

        #
        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # shape: (1, state_dim)

        # Decompose state
        x, xref, uref, t = self.trim_state(state)

        # Safely create leaf tensor for gradient tracking
        xref = xref.requires_grad_()

        # Compute Jacobians inside enable_grad context
        with torch.enable_grad():
            # Compute f and B
            f_xref, B_xref, _ = self.get_f_and_B(xref)
            if len(f_xref.shape) == 1:
                f_xref = f_xref.unsqueeze(0)
            if len(B_xref.shape) == 2:
                B_xref = B_xref.unsqueeze(0)

            DfDx = self.Jacobian(f_xref, xref)  # shape: (1, x_dim, x_dim)
            DBDx = self.B_Jacobian(B_xref, xref)  # shape: (1, x_dim, x_dim, u_dim)

        # Compute A matrix: A = DfDx + sum_j uref_j * dB_j/dx
        A = DfDx.clone().squeeze(0)  # shape: (x_dim, x_dim)
        for j in range(self.action_dim):
            A += uref[0, j] * DBDx[0, :, :, j]  # shape: (x_dim, x_dim)

        B = B_xref.to(self._dtype).squeeze(0)  # shape: (x_dim, u_dim)

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
        e = x - xref  # shape: (1, x_dim)
        u = uref - (K @ e.unsqueeze(-1)).squeeze(-1)

        # Return
        return u, {
            "probs": self.dummy,
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

    def learn(self, batch):
        pass
