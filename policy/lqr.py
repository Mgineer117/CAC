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
        effective_indices: list,
        action_dim: int,
        f_func: Callable,
        B_func: Callable,
        Bbot_func: Callable,
        Q_scaler: float = 1.0,
        R_scaler: float = 1.0,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 1,
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
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices
        self.action_dim = action_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size

        self._forward_steps = 0
        self.nupdates = nupdates
        self.current_update = 0

        self.Q_scaler = Q_scaler
        self.R_scaler = R_scaler
        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func

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
        self._forward_steps += 1
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # shape: (1, state_dim)

        # Decompose state
        x, xref, uref, x_trim, xref_trim = self.trim_state(state)

        # Safely create leaf tensor for gradient tracking
        xref = xref.requires_grad_()

        # Compute Jacobians inside enable_grad context
        with torch.enable_grad():
            # Compute f and B
            f_xref = self.f_func(xref).to(self._dtype)
            B_xref = self.B_func(xref).to(self._dtype)
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
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        loss_dict = {
            f"{self.name}/analytics/avg_rewards": np.mean(batch["rewards"]).item()
        }
        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0
        return loss_dict, timesteps, update_time


class LQR_Approximation(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        action_dim: int,
        Dynamic_func: nn.Module,
        Dynamic_lr: float = 1e-3,
        f_func: Callable | None = None,
        B_func: Callable | None = None,
        Bbot_func: Callable | None = None,
        Q_scaler: float = 1.0,
        R_scaler: float = 1.0,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 1,
        dt: float = 0.03,
        device: str = "cpu",
    ):
        super(LQR_Approximation, self).__init__()

        """
        Do not use Multiprocessor => use less batch
        """
        # constants
        self.name = "LQR_Approximation"
        self.device = device

        self.x_dim = x_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices
        self.action_dim = action_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size

        self.nupdates = nupdates
        self.current_update = 0

        self.Dynamic_func = Dynamic_func
        self.optimizer = torch.optim.Adam(
            params=Dynamic_func.parameters(), lr=Dynamic_lr
        )

        self.Q_scaler = Q_scaler
        self.R_scaler = R_scaler
        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        #
        self.dt = dt
        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

    def lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # shape: (1, state_dim)

        # Decompose state
        x, xref, uref, x_trim, xref_trim = self.trim_state(state)

        if not deterministic:
            u = uref + torch.randn_like(uref) * 0.1
        else:
            xref = xref.requires_grad_()

            # Compute Jacobians inside enable_grad context
            with torch.enable_grad():
                # Compute f and B
                f_xref, B_xref = self.Dynamic_func(xref)
                DfDx = self.Jacobian(f_xref, xref)  # shape: (1, x_dim, x_dim)
                DBDx = self.B_Jacobian(B_xref, xref)  # shape: (1, x_dim, x_dim, u_dim)

            # Compute A matrix: A = DfDx + sum_j uref_j * dB_j/dx
            A = DfDx.clone().squeeze(0)  # shape: (x_dim, x_dim)
            for j in range(self.action_dim):
                A += uref[0, j] * DBDx[0, :, :, j]  # shape: (x_dim, x_dim)

            B = B_xref.squeeze(0)  # shape: (x_dim, u_dim)

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
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        if self.current_update < int(0.95 * self.nupdates):
            # Ingredients: Convert batch data to tensors
            def to_tensor(data):
                return torch.from_numpy(data).to(self._dtype).to(self.device)

            states = to_tensor(batch["states"])
            actions = to_tensor(batch["actions"])

            x, xref, uref, x_trim, xref_trim = self.trim_state(states)

            f = self.f_func(x).to(self._dtype).to(self.device)  # n, x_dim
            B = self.B_func(x).to(self._dtype).to(self.device)  # n, x_dim, action
            dot_x = f + matmul(B, actions.unsqueeze(-1)).squeeze(-1)

            f_approx, B_approx = self.Dynamic_func(x)
            dot_x_approx = f_approx + matmul(B_approx, actions.unsqueeze(-1)).squeeze(
                -1
            )

            fB_loss = F.mse_loss(dot_x, dot_x_approx)

            self.optimizer.zero_grad()
            fB_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
            grad_dict = self.compute_gradient_norm(
                [self.Dynamic_func],
                ["Dynamic_func"],
                dir=f"{self.name}",
                device=self.device,
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            norm_dict = self.compute_weight_norm(
                [self.Dynamic_func],
                ["Dynamic_func"],
                dir=f"{self.name}",
                device=self.device,
            )

            with torch.no_grad():
                f = self.f_func(x).to(self._dtype)
                B = self.B_func(x).to(self._dtype)
                dot_x = f + matmul(B, actions.unsqueeze(-1)).squeeze(-1)

                f_error = F.l1_loss(f, f_approx)
                B_error = F.l1_loss(B, B_approx)
                dot_x_error = F.l1_loss(dot_x, dot_x_approx)

            loss_dict = {
                f"{self.name}/loss/fB_loss": fB_loss.item(),
                f"{self.name}/loss/f_error": f_error.item(),
                f"{self.name}/loss/B_error": B_error.item(),
                f"{self.name}/loss/dot_x_error": dot_x_error.item(),
                f"{self.name}/analytics/Dynamic_lr": self.optimizer.param_groups[0][
                    "lr"
                ],
                f"{self.name}/analytics/avg_rewards": np.mean(batch["rewards"]).item(),
            }
            loss_dict.update(grad_dict)
            loss_dict.update(norm_dict)

            self.current_update += 1
        else:
            loss_dict = {
                f"{self.name}/analytics/avg_rewards": np.mean(batch["rewards"]).item(),
            }
        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0

        return loss_dict, timesteps, update_time
