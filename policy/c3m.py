import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inverse, matmul, transpose
from torch.autograd import grad
from torch.linalg import matrix_norm
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base


class C3M(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        action_dim: int,
        W_func: nn.Module,
        u_func: nn.Module,
        f_func: Callable,
        B_func: Callable,
        Bbot_func: Callable,
        W_lr: float = 3e-4,
        u_lr: float = 3e-4,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        w_ub: float = 1e-2,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 0,
        device: str = "cpu",
    ):
        super(C3M, self).__init__()

        # constants
        self.name = "C3M"
        self.device = device

        self.x_dim = x_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices
        self.action_dim = action_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size

        self.nupdates = nupdates
        self.current_update = 0

        # trainable networks
        self.W_func = W_func
        self.u_func = u_func
        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func
        self.lbd = lbd
        self.eps = eps
        self.w_ub = w_ub

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": self.u_func.parameters(), "lr": u_lr},
            ]
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        #
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
            state = state.unsqueeze(0)

        x, xref, uref, x_trim, xref_trim = self.trim_state(state)
        a, _ = self.u_func(x, xref, uref, x_trim, xref_trim)

        return a, {
            "probs": self.dummy,  # dummy for code consistency
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

    def learn(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        detach = True if self.current_update <= int(0.1 * self.nupdates) else False

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        rewards = to_tensor(batch["rewards"])

        #### COMPUTE INGREDIENTS ####
        # grad tracking state elements
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        W = self.W_func(x, xref, uref, x_trim, xref_trim)  # n, x_dim, x_dim
        M = inverse(W)  # n, x_dim, x_dim

        f = self.f_func(x).to(self._dtype).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self._dtype).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim
        Bbot = (
            self.Bbot_func(x).to(self._dtype).to(self.device)
        )  # n, x_dim, state - action dim

        # since online we do not do below
        u, _ = self.u_func(x, xref, uref, x_trim, xref_trim)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        #  DBDx[:, :, :, i]: n, x_dim, x_dim
        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x, detach)
        dot_W = self.weighted_gradients(W, dot_x, x, detach)

        # contraction condition
        if detach:
            ABK = A + matmul(B, K)
            MABK = matmul(M.detach(), ABK)
            sym_MABK = MABK + transpose(MABK, 1, 2)
            C_u = dot_M + sym_MABK + 2 * self.lbd * M.detach()
        else:
            ABK = A + matmul(B, K)
            MABK = matmul(M, ABK)
            sym_MABK = MABK + transpose(MABK, 1, 2)
            C_u = dot_M + sym_MABK + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f, x, detach)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = DfDxW + transpose(DfDxW, 1, 2)

        # this has to be a negative definite matrix
        C1_inner = -DfW + sym_DfDxW + 2 * self.lbd * W
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)

        C2_inners = []
        C2s = []
        for j in range(self.action_dim):
            DbW = self.weighted_gradients(W, B[:, :, j], x, detach)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = DbDxW + transpose(DbDxW, 1, 2)
            C2_inner = DbW - sym_DbDxW
            C2 = matmul(matmul(transpose(Bbot, 1, 2), C2_inner), Bbot)

            C2_inners.append(C2_inner)
            C2s.append(C2)

        #### COMPUTE LOSS ####
        pd_loss = self.loss_pos_matrix_random_sampling(
            -C_u - self.eps * torch.eye(C_u.shape[-1]).to(self.device)
        )
        c1_loss = self.loss_pos_matrix_random_sampling(
            -C1 - self.eps * torch.eye(C1.shape[-1]).to(self.device)
        )
        # c2_loss = sum([C2.sum().mean() for C2 in C2s])
        c2_loss = sum([(matrix_norm(C2) ** 2).mean() for C2 in C2s])
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )

        loss = pd_loss + c1_loss + c2_loss + overshoot_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir="C3M",
            device=self.device,
        )
        self.optimizer.step()

        with torch.no_grad():
            dot_M_pos_eig, dot_M_neg_eig = self.get_matrix_eig(dot_M)
            sym_MABK_pos_eig, sym_MABK_neg_eig = self.get_matrix_eig(sym_MABK)
            M_pos_eig, M_neg_eig = self.get_matrix_eig(M)

            C_pos_eig, C_neg_eig = self.get_matrix_eig(C_u)
            C1_pos_eig, C1_neg_eig = self.get_matrix_eig(C1)

        # Logging
        loss_dict = {
            "C3M/loss/loss": loss.item(),
            "C3M/loss/pd_loss": pd_loss.item(),
            "C3M/loss/c1_loss": c1_loss.item(),
            "C3M/loss/c2_loss": c2_loss.item(),
            "C3M/loss/overshoot_loss": overshoot_loss.item(),
            "C3M/analytics/C_pos_eig": C_pos_eig.item(),
            "C3M/analytics/C_neg_eig": C_neg_eig.item(),
            "C3M/analytics/C1_pos_eig": C1_pos_eig.item(),
            "C3M/analytics/C1_neg_eig": C1_neg_eig.item(),
            "C3M/analytics/avg_rewards": torch.mean(rewards).item(),
            "C3M/analytics/dot_M_pos_eig": dot_M_pos_eig.item(),
            "C3M/analytics/dot_M_neg_eig": dot_M_neg_eig.item(),
            "C3M/analytics/sym_MABK_pos_eig": sym_MABK_pos_eig.item(),
            "C3M/analytics/sym_MABK_neg_eig": sym_MABK_neg_eig.item(),
            "C3M/analytics/M_pos_eig": M_pos_eig.item(),
            "C3M/analytics/M_neg_eig": M_neg_eig.item(),
        }
        norm_dict = self.compute_weight_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir="C3M",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, u, rewards
        self.eval()
        self.current_update += 1

        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0
        self.lr_scheduler.step()

        return loss_dict, timesteps, update_time


class C3M_Approximation(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        action_dim: int,
        W_func: nn.Module,
        u_func: nn.Module,
        Dynamic_func: nn.Module,
        f_func: Callable,
        B_func: Callable,
        Bbot_func: Callable,
        W_lr: float = 3e-4,
        u_lr: float = 3e-4,
        Dynamic_lr: float = 3e-4,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        w_ub: float = 1e-2,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 0,
        dt: float = 0.03,
        device: str = "cpu",
    ):
        super(C3M_Approximation, self).__init__()

        # constants
        self.name = "C3M_Approximation"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = action_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size

        self.eps = eps
        self.lbd = lbd
        self.w_ub = w_ub
        self.dt = dt

        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func

        self.nupdates = nupdates
        self.num_outer_update = 0
        self.num_inner_update = 0

        # trainable networks
        self.W_func = W_func
        self.u_func = u_func
        self.Dynamic_func = Dynamic_func

        self.W_u_optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": self.u_func.parameters(), "lr": u_lr},
            ]
        )
        self.Dynamic_optimizer = torch.optim.Adam(
            params=self.Dynamic_func.parameters(), lr=Dynamic_lr
        )

        self.W_lr_scheduler = LambdaLR(self.W_u_optimizer, lr_lambda=self.W_lr_fn)
        self.D_lr_scheduler = LambdaLR(self.Dynamic_optimizer, lr_lambda=self.D_lr_fn)

        #
        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

    def W_lr_fn(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def D_lr_fn(self, step):
        return 0.999**step

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        x, xref, uref, x_trim, xref_trim = self.trim_state(state)
        a, _ = self.u_func(x, xref, uref, x_trim, xref_trim)

        return a, {
            "probs": self.dummy,
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

    def contraction_loss(
        self,
        states: torch.Tensor,
        detach: bool,
    ):
        true_dict = self.get_true_metrics(states)

        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        W = self.W_func(x, xref, uref, x_trim, xref_trim)
        M = inverse(W)

        f_approx, B_approx = self.Dynamic_func(x)
        Bbot_approx = self.compute_B_perp_batch(B_approx, self.x_dim - self.action_dim)

        DfDx = self.Jacobian(f_approx, x).detach()  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B_approx, x).detach()  # n, x_dim, x_dim, b_dim

        f_approx = f_approx.detach()
        B_approx = B_approx.detach()
        Bbot_approx = Bbot_approx.detach()

        u, _ = self.u_func(x, xref, uref, x_trim, xref_trim)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        # contraction condition
        dot_M_approx = self.weighted_gradients(M, true_dict["dot_x_true"], x, detach)
        ABK_approx = A + matmul(B_approx, K)
        if detach:
            MABK_approx = matmul(M.detach(), ABK_approx)
            sym_MABK_approx = MABK_approx + transpose(MABK_approx, 1, 2)
            C_u = dot_M_approx + sym_MABK_approx + 2 * self.lbd * M.detach()
        else:
            MABK_approx = matmul(M, ABK_approx)
            sym_MABK_approx = MABK_approx + transpose(MABK_approx, 1, 2)
            C_u = dot_M_approx + sym_MABK_approx + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f_approx, x, detach)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = DfDxW + transpose(DfDxW, 1, 2)

        # this has to be a negative definite matrix
        C1_inner = -DfW + sym_DfDxW + 2 * self.lbd * W
        C1 = matmul(matmul(transpose(Bbot_approx, 1, 2), C1_inner), Bbot_approx)

        C2_inners = []
        C2s = []
        for j in range(self.action_dim):
            DbW = self.weighted_gradients(W, B_approx[:, :, j], x, detach)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = DbDxW + transpose(DbDxW, 1, 2)
            C2_inner = DbW - sym_DbDxW
            C2 = matmul(matmul(transpose(Bbot_approx, 1, 2), C2_inner), Bbot_approx)
            C2_inners.append(C2_inner)
            C2s.append(C2)

        #### COMPUTE LOSS ####
        pd_loss = self.loss_pos_matrix_random_sampling(
            -C_u - self.eps * torch.eye(C_u.shape[-1]).to(self.device)
        )
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )
        c1_loss = self.loss_pos_matrix_random_sampling(
            -C1 - self.eps * torch.eye(C1.shape[-1]).to(self.device)
        )
        # c2_loss = sum([C2.sum().mean() for C2 in C2s])
        c2_loss = sum([(matrix_norm(C2) ** 2).mean() for C2 in C2s])

        loss = pd_loss + overshoot_loss + c1_loss + c2_loss

        ### for loggings
        with torch.no_grad():
            dot_M_pos_eig, dot_M_neg_eig = self.get_matrix_eig(true_dict["dot_M_true"])
            sym_MABK_pos_eig, sym_MABK_neg_eig = self.get_matrix_eig(
                true_dict["sym_MABK_true"]
            )
            M_pos_eig, M_neg_eig = self.get_matrix_eig(M)

            C_pos_eig, C_neg_eig = self.get_matrix_eig(C_u)
            C1_pos_eig, C1_neg_eig = self.get_matrix_eig(C1)

            dot_M_error = matrix_norm(
                true_dict["dot_M_true"] - dot_M_approx, ord="fro"
            ).mean()
            ABK_error = matrix_norm(
                true_dict["ABK_true"] - ABK_approx, ord="fro"
            ).mean()
            Bbot_error = matrix_norm(
                true_dict["Bbot_true"] - Bbot_approx, ord="fro"
            ).mean()

        return (
            loss,
            {
                "pd_loss": pd_loss.item(),
                "overshoot_loss": overshoot_loss.item(),
                "c1_loss": c1_loss.item(),
                "c2_loss": c2_loss.item(),
                "C_pos_eig": C_pos_eig.item(),
                "C_neg_eig": C_neg_eig.item(),
                "C1_pos_eig": C1_pos_eig.item(),
                "C1_neg_eig": C1_neg_eig.item(),
                "dot_M_pos_eig": dot_M_pos_eig.item(),
                "dot_M_neg_eig": dot_M_neg_eig.item(),
                "sym_MABK_pos_eig": sym_MABK_pos_eig.item(),
                "sym_MABK_neg_eig": sym_MABK_neg_eig.item(),
                "M_pos_eig": M_pos_eig.item(),
                "M_neg_eig": M_neg_eig.item(),
                "dot_M_error": dot_M_error.item(),
                "ABK_error": ABK_error.item(),
                "Bbot_error": Bbot_error.item(),
            },
        )

    def learn(self, batch):
        if self.num_inner_update <= int(0.05 * self.nupdates):
            loss_dict, update_time = self.learn_Dynamics(batch)
            loss_dict = {}
            timesteps = 0
            update_time = 0
            # timesteps = batch["rewards"].shape[0]
            self.num_inner_update += 1
        else:
            detach = (
                True if self.num_outer_update <= int(0.1 * self.nupdates) else False
            )

            loss_dict, timesteps, update_time = self.learn_W(batch, detach)
            D_loss_dict, D_update_time = self.learn_Dynamics(batch)

            loss_dict.update(D_loss_dict)
            update_time += D_update_time

            self.num_outer_update += 1
            self.W_lr_scheduler.step()
            self.D_lr_scheduler.step()

            self.num_outer_update += 1

        return loss_dict, timesteps, update_time

    def learn_Dynamics(self, batch: dict):
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])

        true_dict = self.get_true_metrics(states)

        x, _, _, _, _ = self.trim_state(states)
        f_approx, B_approx = self.Dynamic_func(x)

        f = self.f_func(x).to(self._dtype).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self._dtype).to(self.device)  # n, x_dim, action

        dot_x_true = f + matmul(B, actions.unsqueeze(-1)).squeeze(-1)
        dot_x_approx = f_approx + matmul(B_approx, actions.unsqueeze(-1)).squeeze(-1)

        loss = F.mse_loss(dot_x_true, dot_x_approx)

        with torch.no_grad():
            f_error = F.l1_loss(true_dict["f_true"], f_approx)
            B_error = F.l1_loss(true_dict["B_true"], B_approx)

        self.Dynamic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Dynamic_func.parameters(), max_norm=5.0)
        grad_dict = self.compute_gradient_norm(
            [self.Dynamic_func],
            ["Dynamic_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.Dynamic_optimizer.step()

        norm_dict = self.compute_weight_norm(
            [self.Dynamic_func],
            ["Dynamic_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict = {
            f"{self.name}/Dynamic_loss/loss": loss.item(),
            f"{self.name}/Dynamic_analytics/f_error": f_error.item(),
            f"{self.name}/Dynamic_analytics/B_error": B_error.item(),
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, x
        self.eval()

        update_time = time.time() - t0
        return loss_dict, update_time

    def learn_W(self, batch: dict, detach: bool):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])
        next_states = to_tensor(batch["next_states"])
        rewards = to_tensor(batch["rewards"])
        terminals = to_tensor(batch["terminals"])

        # List to track actor loss over minibatches
        loss, infos = self.contraction_loss(states, detach)

        self.W_u_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.W_func.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.u_func.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.W_u_optimizer.step()
        norm_dict = self.compute_weight_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir=f"{self.name}",
            device=self.device,
        )

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/pd_loss": infos["pd_loss"],
            f"{self.name}/loss/overshoot_loss": infos["overshoot_loss"],
            f"{self.name}/loss/c1_loss": infos["c1_loss"],
            f"{self.name}/loss/c2_loss": infos["c2_loss"],
            f"{self.name}/C_analytics/C_pos_eig": infos["C_pos_eig"],
            f"{self.name}/C_analytics/C_neg_eig": infos["C_neg_eig"],
            f"{self.name}/C_analytics/C1_pos_eig": infos["C1_pos_eig"],
            f"{self.name}/C_analytics/C1_neg_eig": infos["C1_neg_eig"],
            f"{self.name}/C_analytics/dot_M_pos_eig": infos["dot_M_pos_eig"],
            f"{self.name}/C_analytics/dot_M_neg_eig": infos["dot_M_neg_eig"],
            f"{self.name}/C_analytics/sym_MABK_pos_eig": infos["sym_MABK_pos_eig"],
            f"{self.name}/C_analytics/sym_MABK_neg_eig": infos["sym_MABK_neg_eig"],
            f"{self.name}/C_analytics/M_pos_eig": infos["M_pos_eig"],
            f"{self.name}/C_analytics/M_neg_eig": infos["M_neg_eig"],
            f"{self.name}/C_analytics/dot_M_error": infos["dot_M_error"],
            f"{self.name}/C_analytics/ABK_error": infos["ABK_error"],
            f"{self.name}/C_analytics/Bbot_error": infos["Bbot_error"],
            f"{self.name}/analytics/avg_rewards": torch.mean(rewards).item(),
        }

        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        timesteps = terminals.shape[0]

        # Cleanup
        del states, actions, next_states, terminals
        self.eval()

        update_time = time.time() - t0

        return loss_dict, timesteps, update_time

    def get_true_metrics(self, states: torch.Tensor):
        #### COMPUTE THE REAL DYNAMICS TO MEASURE ERRORS ####
        states = states.requires_grad_()
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)

        with torch.no_grad():
            W = self.W_func(x, xref, uref, x_trim, xref_trim)
            M = inverse(W)

        f = self.f_func(x).to(self._dtype).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self._dtype).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x).detach()  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x).detach()  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x).detach().to(self._dtype).to(self.device)

        u, _ = self.u_func(x, xref, uref, x_trim, xref_trim)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        dot_x = (f + matmul(B, u.unsqueeze(-1)).squeeze(-1)).detach()
        dot_M = self.weighted_gradients(M, dot_x, x, True)

        ABK = A + matmul(B, K)
        MABK = matmul(M.detach(), ABK)
        sym_MABK = MABK + transpose(MABK, 1, 2)

        return {
            "dot_x_true": dot_x,
            "dot_M_true": dot_M,
            "ABK_true": ABK,
            "sym_MABK_true": sym_MABK,
            "Bbot_true": Bbot,
            "f_true": f,
            "B_true": B,
        }
