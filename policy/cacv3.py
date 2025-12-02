import time
from copy import deepcopy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inverse, matmul, transpose
from torch.linalg import matrix_norm
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base
from policy.cac import CAC
from utils.functions import (
    compute_kl,
    conjugate_gradients,
    estimate_advantages,
    flat_params,
    hessian_vector_product,
    set_flat_params,
)


class CACv3(CAC):
    def __init__(
        self,
        # Learning parameters
        x_dim: int,
        data: dict,
        W_func: nn.Module,
        get_f_and_B: Callable,
        actor: nn.Module,
        critic: nn.Module,
        W_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        # CMG parameters
        w_ub: float = 10.0,
        w_lb: float = 1e-1,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        W_entropy_scaler: float = 1e-3,
        # TRPO parameters
        damping: float = 1e-1,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        target_kl: float = 0.03,
        # PPO parameters
        eps_clip: float = 0.2,
        K: int = 5,
        entropy_scaler: float = 1e-3,
        # RL parameters
        gamma: float = 0.99,
        gae: float = 0.95,
        l2_reg: float = 1e-8,
        tracking_scaler: float = 1.0,
        control_scaler: float = 0.0,
        nupdates: int = 1,
        device: str = "cpu",
    ):
        super(CACv3, self).__init__(
            x_dim=x_dim,
            data=data,
            W_func=W_func,
            get_f_and_B=get_f_and_B,
            actor=actor,
            critic=critic,
            W_lr=W_lr,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            num_minibatch=num_minibatch,
            minibatch_size=minibatch_size,
            w_ub=w_ub,
            w_lb=w_lb,
            lbd=lbd,
            eps=eps,
            W_entropy_scaler=W_entropy_scaler,
            damping=damping,
            backtrack_iters=backtrack_iters,
            backtrack_coeff=backtrack_coeff,
            target_kl=target_kl,
            eps_clip=eps_clip,
            K=K,
            entropy_scaler=entropy_scaler,
            gamma=gamma,
            gae=gae,
            l2_reg=l2_reg,
            tracking_scaler=tracking_scaler,
            control_scaler=control_scaler,
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

        self.W_optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": [self.lbd], "lr": 1e-4},
                {"params": [self.w_ub], "lr": 1e-4},
                {"params": [self.w_lb], "lr": 1e-4},
            ]
        )
        self.dual_optimizer = torch.optim.Adam(
            [{"params": [self.nu], "lr": 1e-2}, {"params": [self.zeta], "lr": 1e-2}]
        )

        self.lr_scheduler1 = LambdaLR(self.W_optimizer, lr_lambda=self.lr_lambda)
        self.lr_scheduler3 = LambdaLR(self.dual_optimizer, lr_lambda=self.lr_lambda)

        self.to(self._dtype).to(self.device)

    def compute_W_loss(self):
        #
        I = torch.eye(self.x_dim, device=self.device)

        # === SAMPLE BATCH === #
        batch = dict()
        buffer_size, batch_size = self.data["x"].shape[0], 1024
        indices = np.random.choice(buffer_size, size=batch_size, replace=False)
        for key in self.data.keys():
            # Sample a batch of 1024
            batch[key] = self.data[key][indices]

        # === PREPARE TENSORS === #
        x = self.to_tensor(batch["x"]).requires_grad_()
        xref = self.to_tensor(batch["xref"])
        uref = self.to_tensor(batch["uref"])

        raw_W, _ = self.W_func(x)  # n, x_dim, x_dim
        # Add lower-bound scaled identity to guarantee positive definiteness
        W = raw_W + self.w_lb * I
        M = inverse(W)  # n, x_dim, x_dim

        f, B, Bbot = self.get_f_and_B(x)
        f = f.to(self._dtype).to(self.device)  # n, x_dim
        B = B.to(self._dtype).to(self.device)  # n, x_dim, action
        Bbot = Bbot.to(self._dtype).to(self.device)  #

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim

        f = f.detach()
        B = B.detach()
        Bbot = Bbot.detach()

        # since online we do not do below
        u, _ = self.actor(x, xref, uref)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        # detach actor gradients
        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x)

        # contraction condition
        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
        Cu = dot_M + sym_MABK + 2 * self.lbd * M.detach()

        # C1
        DfW = self.weighted_gradients(W, f, x)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = 0.5 * (DfDxW + transpose(DfDxW, 1, 2))

        # this has to be a negative definite matrix
        C1_inner = -DfW + sym_DfDxW + 2 * self.lbd * W.detach()
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)

        C2_inners = []
        C2s = []
        for j in range(self.action_dim):
            DbW = self.weighted_gradients(W, B[:, :, j], x)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = DbDxW + transpose(DbDxW, 1, 2)
            C2_inner = DbW - sym_DbDxW
            C2 = matmul(matmul(transpose(Bbot, 1, 2), C2_inner), Bbot)

            C2_inners.append(C2_inner)
            C2s.append(C2)

        ### DEFINE PD MATRICES ###
        Cu = Cu + self.eps * torch.eye(Cu.shape[-1], device=self.device)
        C1 = C1 + self.eps * torch.eye(C1.shape[-1], device=self.device)
        C2 = sum([(C2**2).reshape(batch_size, -1).sum(1).mean() for C2 in C2s])
        overshoot = W - self.w_ub * I

        # === DEFINE LOSSES === #
        pd_loss, pd_reg = self.loss_pos_matrix_random_sampling(-Cu)
        c1_loss, c1_reg = self.loss_pos_matrix_random_sampling(-C1)
        overshoot_loss, overshoot_reg = self.loss_pos_matrix_random_sampling(-overshoot)
        c2_loss = C2
        self.record_eigenvalues(Cu, dot_M, sym_MABK, C1, C2, overshoot)

        primal_loss = (
            (1 / self.lbd) ** 2 * (self.w_ub / self.w_lb)
            + self.nu[0].detach() * overshoot_loss
            + self.nu[1].detach() * pd_loss
            + self.nu[2].detach() * c1_loss
            + self.zeta.detach() * c2_loss
            + pd_reg
            + c1_reg
            + overshoot_reg
        )

        dual_loss = -(
            self.nu[0] * overshoot_loss.detach()
            + self.nu[1] * pd_loss.detach()
            + self.nu[2] * c1_loss.detach()
            + self.zeta * c2_loss.detach()
        )

        return (
            primal_loss,
            dual_loss,
            {
                "pd_loss": pd_loss,
                "c1_loss": c1_loss,
                "c2_loss": c2_loss,
                "overshoot_loss": overshoot_loss,
            },
        )

    def optimize_W_params(self, primal_loss: torch.Tensor, dual_loss: torch.Tensor):
        # === OPTIMIZATION STEP === #
        self.W_optimizer.zero_grad()
        primal_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func, self.lbd],
            ["W_func", "lbd"],
            dir="CAC",
            device=self.device,
        )
        self.W_optimizer.step()

        self.dual_optimizer.zero_grad()
        dual_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.dual_optimizer.param_groups[0]["params"], max_norm=10.0
        )
        grad_dict.update(
            self.compute_gradient_norm(
                [self.nu, self.zeta],
                ["nu", "zeta"],
                dir=f"{self.name}",
                device=self.device,
            )
        )
        self.dual_optimizer.step()

        # ensure the primal and dual feasibility
        with torch.no_grad():
            self.lbd.clamp_(min=1e-6, max=1e6)
            self.nu.clamp_(min=0.0, max=1e6)

        return grad_dict

    def learn(self, batch):
        loss_dict, supp_dict = {}, {}

        W_update_time = 0
        if self.num_RL_updates % 3 == 0:
            W_loss_dict, W_supp_dict, W_update_time = self.learn_W()
            loss_dict.update(W_loss_dict)
            supp_dict.update(W_supp_dict)

        RL_loss_dict, RL_supp_dict, RL_update_time = self.learn_ppo(batch)
        # RL_loss_dict, RL_supp_dict, RL_update_time = self.learn_trpo(batch)
        loss_dict.update(RL_loss_dict)
        supp_dict.update(RL_supp_dict)

        self.lr_scheduler1.step()
        self.lr_scheduler2.step()
        self.lr_scheduler3.step()

        update_time = W_update_time + RL_update_time
        self.num_RL_updates += 1

        return loss_dict, supp_dict, update_time

    def learn_W(self):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # === PERFORM OPTIMIZATION STEP === #
        primal_loss, dual_loss, infos = self.compute_W_loss()
        grad_dict = self.optimize_W_params(primal_loss, dual_loss)

        # === LOGGING === #
        supp_dict = {}
        if self.num_W_updates % 300 == 0:
            fig = self.get_eigenvalue_plot()
            supp_dict["CAC/plot/eigenvalues"] = fig

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
            f"{self.name}/lr/lbd_lr": self.lr_scheduler1.get_last_lr()[1],
            f"{self.name}/lr/nu_lr": self.lr_scheduler3.get_last_lr()[0],
            f"{self.name}/lr/zeta_lr": self.lr_scheduler3.get_last_lr()[1],
        }
        norm_dict = self.compute_weight_norm(
            [self.W_func],
            ["W_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        self.eval()
        update_time = time.time() - t0
        self.num_W_updates += 1

        return loss_dict, supp_dict, update_time
