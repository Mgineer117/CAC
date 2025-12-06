import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import inverse, matmul, transpose
from torch.optim.lr_scheduler import LambdaLR

from policy.c3m import C3M


class C3Mv3(C3M):
    def __init__(
        self,
        x_dim: int,
        action_dim: int,
        W_func: nn.Module,
        u_func: nn.Module,
        data: dict,
        get_f_and_B: Callable,
        W_lr: float = 3e-4,
        u_lr: float = 3e-4,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        w_ub: float = 10.0,
        w_lb: float = 1e-1,
        gamma: float = 0.99,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 1,
        device: str = "cpu",
    ):
        super(C3Mv3, self).__init__(
            x_dim=x_dim,
            action_dim=action_dim,
            W_func=W_func,
            u_func=u_func,
            data=data,
            get_f_and_B=get_f_and_B,
            W_lr=W_lr,
            u_lr=u_lr,
            lbd=lbd,
            eps=eps,
            w_ub=w_ub,
            w_lb=w_lb,
            gamma=gamma,
            num_minibatch=num_minibatch,
            minibatch_size=minibatch_size,
            nupdates=nupdates,
            device=device,
        )

        # make lbd and nu a trainable parameter
        self.lbd = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, device=self.device) + 1e-2
        )
        self.w_ub = nn.Parameter(
            torch.tensor(w_ub, dtype=torch.float32, device=self.device)
        )
        self.w_lb = nn.Parameter(
            torch.tensor(w_lb, dtype=torch.float32, device=self.device)
        )
        self.nu = nn.Parameter(
            torch.ones(3, dtype=torch.float32, device=self.device) + 1e-2
        )
        self.zeta = nn.Parameter(
            torch.ones(1, dtype=torch.float32, device=self.device) + 1e-2
        )

        self.primal_optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": self.u_func.parameters(), "lr": u_lr},
                {"params": [self.lbd], "lr": W_lr},
                {"params": [self.w_ub], "lr": 1e-4},
                {"params": [self.w_lb], "lr": 1e-4},
            ]
        )
        self.dual_optimizer = torch.optim.Adam(
            [{"params": [self.nu], "lr": 1e-2}, {"params": [self.zeta], "lr": 1e-2}]
        )

        self.lr_scheduler1 = LambdaLR(self.primal_optimizer, lr_lambda=self.lr_lambda)
        self.lr_scheduler2 = LambdaLR(self.dual_optimizer, lr_lambda=self.lr_lambda)

        #
        self.num_updates = 0
        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

    def compute_loss(self):
        # 1. Prepare Data
        x, xref, uref = self._sample_batch(batch_size=1024)
        Ix = torch.eye(self.x_dim, device=self.device)
        Iu = torch.eye(self.x_dim - self.action_dim, device=self.device)

        # 2. Get System Matrices & Metric
        raw_W, _ = self.W_func(x)

        # Get physics terms and strictly move to GPU/Correct Dtype
        f, B, Bbot = self.get_f_and_B(x)
        f = f.to(dtype=self._dtype, device=self.device)
        B = B.to(dtype=self._dtype, device=self.device)
        Bbot = Bbot.to(dtype=self._dtype, device=self.device)

        f, B, Bbot = f.detach(), B.detach(), Bbot.detach()

        # 3. Compute Derivatives & Controller
        u, _ = self.u_func(x, xref, uref)

        # 4. Compute Constraints (The heavy math is moved to helpers)
        # -- L_u (Contraction)
        Cu, dot_M, sym_MABK = self._compute_contraction_condition(
            x, u, f, B, raw_W, self.lbd
        )

        # -- L_w1 (Feasibility on perp manifold)
        C1 = self._compute_C1_condition(x, f, Bbot, raw_W, self.lbd)

        # -- L_w2 (Matching condition)
        C2_val, C2s = self._compute_C2_condition(x, B, Bbot, raw_W)

        # -- Bounds
        overshoot = raw_W - self.w_ub * Ix + self.w_lb.detach() * Ix

        # 5. Calculate Losses (using sampling helper)
        pd_loss, pd_reg = self.loss_pos_matrix_random_sampling(-(Cu + self.eps * Ix))
        c1_loss, c1_reg = self.loss_pos_matrix_random_sampling(-(C1 + self.eps * Iu))
        overshoot_loss, overshoot_reg = self.loss_pos_matrix_random_sampling(-overshoot)
        c2_loss = C2_val

        # Logging
        self.record_eigenvalues(Cu, dot_M, sym_MABK, C1, C2_val, raw_W)

        # === COMBINE LOSSES === #
        if not self.warmup_complete:
            primal_loss = c1_loss + c1_reg + c2_loss
            dual_loss = torch.tensor(0.0, device=self.device)
        else:
            primal_loss = (
                (1 / self.lbd) ** 2 * (self.w_ub / self.w_lb)
                + (self.nu[0].detach() * overshoot_loss + overshoot_reg)
                + (self.nu[1].detach() * pd_loss + pd_reg)
                + (self.nu[2].detach() * c1_loss + c1_reg)
                + (self.zeta.detach() * c2_loss)
            )
            dual_loss = -(
                self.nu[0] * overshoot_loss.detach()
                + self.nu[1] * pd_loss.detach()
                + self.nu[2] * c1_loss.detach()
                + self.zeta * c2_loss.detach()
            )

        loss_dict = {
            "pd_loss": pd_loss,
            "c1_loss": c1_loss,
            "c2_loss": c2_loss,
            "overshoot_loss": overshoot_loss,
        }
        return primal_loss, dual_loss, loss_dict

    def optimize_params(self, primal_loss, dual_loss):
        grad_dict = {}

        # 1. Primal Step
        primal_params = [self.W_func, self.u_func, self.lbd, self.w_ub, self.w_lb]
        primal_names = ["W_func", "u_func", "lbd", "w_ub", "w_lb"]

        # If warmup is done, we update the Lagrange multipliers and bounds
        freeze_list = []
        if not self.warmup_complete:
            freeze_list = [self.lbd, self.w_lb, self.w_ub]

        grad_dict.update(
            self._step_optimizer(
                optimizer=self.primal_optimizer,
                loss=primal_loss,
                params_to_log=primal_params,
                names_to_log=primal_names,
                scheduler=self.lr_scheduler1,
                params_to_freeze=freeze_list,
            )
        )

        # 2. Dual Step (Only if warmup complete)
        if self.warmup_complete:
            grad_dict.update(
                self._step_optimizer(
                    optimizer=self.dual_optimizer,
                    loss=dual_loss,
                    params_to_log=[self.nu, self.zeta],
                    names_to_log=["nu", "zeta"],
                    scheduler=self.lr_scheduler2,
                    params_to_freeze=[],  # No freezing in dual step
                )
            )

            # 3. Feasibility Clamping
            with torch.no_grad():
                self.lbd.clamp_(min=1e-3, max=1e6)
                self.nu.clamp_(min=0.0, max=1e6)
                self.w_lb.clamp_(min=1e-3, max=90.0)
                self.w_ub.clamp_(min=self.w_lb.detach(), max=100.0)

        return grad_dict

    def _sample_batch(self, batch_size):
        # Concise sampling logic
        idxs = np.random.choice(self.data["x"].shape[0], size=batch_size, replace=False)
        return (
            self.to_tensor(self.data["x"][idxs]).requires_grad_(),
            self.to_tensor(self.data["xref"][idxs]),
            self.to_tensor(self.data["uref"][idxs]),
        )

    def _step_optimizer(
        self,
        optimizer,
        loss,
        params_to_log,
        names_to_log,
        scheduler=None,
        params_to_freeze=None,
    ):
        """Generic wrapper for zero_grad -> backward -> freeze -> clip -> step"""
        optimizer.zero_grad()
        loss.backward()

        # === FREEZE SPECIFIC PARAMS === #
        if params_to_freeze:
            for p in params_to_freeze:
                p.grad = None

        # Gather all params for clipping
        all_params = [
            p
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if all_params:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)

        # Compute norms for logging
        grads = self.compute_gradient_norm(
            params_to_log, names_to_log, dir=self.name, device=self.device
        )

        optimizer.step()
        if scheduler:
            scheduler.step()
        return grads

    def _compute_contraction_condition(self, x, u, f, B, raw_W, lbd):
        W = raw_W + self.w_lb * torch.eye(self.x_dim, device=self.device)
        M = inverse(W)

        # Calculate Jacobians
        DfDx = self.Jacobian(f, x)
        DBDx = self.B_Jacobian(B, x)
        K = self.Jacobian(u, x)

        # Dynamics A matrix: df/dx + sum(u_i * db_i/dx)
        sum_u_DBDx = sum(
            [u[:, i, None, None] * DBDx[:, :, :, i] for i in range(self.action_dim)]
        )
        A = DfDx + sum_u_DBDx

        # Total time derivative of M
        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x)

        # Stability condition: dot_M + 2sym(M(A+BK)) + 2lambda*M
        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))

        Cu = dot_M + 2 * sym_MABK + 2 * abs(lbd) * M.detach()
        return Cu, dot_M, sym_MABK

    def _compute_C1_condition(self, x, f, Bbot, raw_W, lbd):
        W = raw_W + self.w_lb.detach() * torch.eye(self.x_dim, device=self.device)

        DfDx = self.Jacobian(f, x)
        DfW = self.weighted_gradients(W, f, x)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = 0.5 * (DfDxW + transpose(DfDxW, 1, 2))

        # Condition 2 (Stability on null space)
        # Bbot.T * (-dot_W + 2sym(A*W) + 2lambda*W) * Bbot
        C1_inner = -DfW + 2 * sym_DfDxW + 2 * abs(lbd.detach()) * W.detach()
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)
        return C1

    def _compute_C2_condition(self, x, B, Bbot, raw_W):
        W = raw_W + self.w_lb.detach() * torch.eye(self.x_dim, device=self.device)

        DBDx = self.B_Jacobian(B, x)
        C2s = []
        for j in range(self.action_dim):
            # Lie derivative along control directions
            DbW = self.weighted_gradients(W, B[:, :, j], x)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = 0.5 * (DbDxW + transpose(DbDxW, 1, 2))

            C2_inner = DbW - 2 * sym_DbDxW
            C2_proj = matmul(matmul(transpose(Bbot, 1, 2), C2_inner), Bbot)
            C2s.append(C2_proj)

        # Sum of squares of norms
        batch_size = x.shape[0]
        C2_total = sum([(C**2).reshape(batch_size, -1).sum(1).mean() for C in C2s])
        return C2_total, C2s

    def learn(self):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Define progress threshold (using 0.1 based on your snippet, or 1.0 if strictly intended)
        self.warmup_complete = (self.num_updates / self.nupdates) >= 0.1

        # === PERFORM OPTIMIZATION STEP === #
        primal_loss, dual_loss, infos = self.compute_loss()
        grad_dict = self.optimize_params(primal_loss, dual_loss)

        # === LOGGING === #
        supp_dict = {}
        if self.num_updates % 500 == 0:
            fig = self.get_eigenvalue_plot()
            supp_dict["C3M/plot/eigenvalues"] = fig

        loss_dict = {
            f"{self.name}/loss/loss": primal_loss.item(),
            f"{self.name}/loss/pd_loss": infos["pd_loss"].item(),
            f"{self.name}/loss/c1_loss": infos["c1_loss"].item(),
            f"{self.name}/loss/c2_loss": infos["c2_loss"].item(),
            f"{self.name}/loss/overshoot_loss": infos["overshoot_loss"].item(),
            f"{self.name}/loss/dual_loss": dual_loss.item(),
            f"{self.name}/analytics/lbd": self.lbd.item(),
            f"{self.name}/analytics/w_ub": self.w_ub.item(),
            f"{self.name}/analytics/w_lb": self.w_lb.item(),
            f"{self.name}/analytics/nu1": self.nu[0].item(),
            f"{self.name}/analytics/nu2": self.nu[1].item(),
            f"{self.name}/analytics/nu3": self.nu[2].item(),
            f"{self.name}/analytics/zeta": self.zeta.item(),
            f"{self.name}/lr/W_lr": self.lr_scheduler1.get_last_lr()[0],
            f"{self.name}/lr/u_lr": self.lr_scheduler1.get_last_lr()[1],
            f"{self.name}/lr/lbd_lr": self.lr_scheduler1.get_last_lr()[2],
            f"{self.name}/lr/w_ub_lr": self.lr_scheduler1.get_last_lr()[3],
            f"{self.name}/lr/w_lb_lr": self.lr_scheduler1.get_last_lr()[4],
            f"{self.name}/lr/nu_lr": self.lr_scheduler2.get_last_lr()[0],
            f"{self.name}/lr/zeta_lr": self.lr_scheduler2.get_last_lr()[1],
        }
        norm_dict = self.compute_weight_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # === CLEANUP === #
        self.eval()
        update_time = time.time() - t0
        self.num_updates += 1

        return loss_dict, supp_dict, update_time
