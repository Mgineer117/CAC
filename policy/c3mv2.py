import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import inverse, matmul, transpose
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base
from policy.c3m import C3M


class C3Mv2(C3M):
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
        w_ub: float = 1e-2,
        gamma: float = 0.99,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 1,
        device: str = "cpu",
    ):
        super(C3Mv2, self).__init__(
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
            gamma=gamma,
            num_minibatch=num_minibatch,
            minibatch_size=minibatch_size,
            nupdates=nupdates,
            device=device,
        )

        # make lbd and nu a trainable parameter
        self.lbd = nn.Parameter(
            torch.tensor(lbd, dtype=torch.float32, device=self.device)
        )
        self.nu = nn.Parameter(
            torch.ones(3, dtype=torch.float32, device=self.device) + 1e-2
        )
        self.zeta = nn.Parameter(
            torch.ones(1, dtype=torch.float32, device=self.device) + 1e-2
        )

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": self.u_func.parameters(), "lr": u_lr},
                {"params": [self.lbd], "lr": 3e-3},
            ]
        )
        self.dual_optimizer = torch.optim.Adam(
            [{"params": [self.nu], "lr": 1e-2}, {"params": [self.zeta], "lr": 1e-2}]
        )

        self.lr_scheduler1 = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        self.lr_scheduler2 = LambdaLR(self.dual_optimizer, lr_lambda=self.lr_lambda)

        #
        self.num_updates = 0
        self.dummy = torch.tensor(1e-5)
        self.to(self._dtype).to(self.device)

    def compute_loss(self):
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

        W, _ = self.W_func(x)  # n, x_dim, x_dim
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
        u, _ = self.u_func(x, xref, uref)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

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
        Cu = dot_M + sym_MABK + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f, x)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = 0.5 * (DfDxW + transpose(DfDxW, 1, 2))

        # this has to be a negative definite matrix
        C1_inner = -DfW + sym_DfDxW + 2 * self.lbd * W
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
        Cu = Cu + self.eps * torch.eye(Cu.shape[-1]).to(self.device)
        C1 = C1 + self.eps * torch.eye(C1.shape[-1]).to(self.device)
        C2 = sum([(C2**2).reshape(batch_size, -1).sum(1).mean() for C2 in C2s])
        overshoot = W - (self.w_ub * torch.eye(W.shape[-1])).unsqueeze(0).to(
            self.device
        )

        # === DEFINE LOSSES === #
        pd_loss = self.loss_pos_matrix_random_sampling(-Cu)
        c1_loss = self.loss_pos_matrix_random_sampling(-C1)
        overshoot_loss = self.loss_pos_matrix_random_sampling(-overshoot)
        c2_loss = C2
        self.record_eigenvalues(Cu, dot_M, sym_MABK, C1, C2, overshoot)

        nu = self.nu.detach()
        zeta = self.zeta.detach()
        primal_loss = (
            -self.lbd
            + nu[0] * overshoot_loss
            + nu[1] * pd_loss
            + nu[2] * c1_loss
            + zeta * c2_loss
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

    def optimize_params(self, primal_loss: torch.Tensor, dual_loss: torch.Tensor):
        # === OPTIMIZATION STEP === #
        self.optimizer.zero_grad()
        primal_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func, self.u_func, self.lbd],
            ["W_func", "u_func", "lbd"],
            dir="C3M",
            device=self.device,
        )
        self.optimizer.step()

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

        # lr scheduling
        self.lr_scheduler1.step()
        self.lr_scheduler2.step()

        # ensure the primal and dual feasibility
        with torch.no_grad():
            self.lbd.clamp_(min=1e-6, max=1e6)
            self.nu.clamp_(min=0.0, max=1e6)

        return grad_dict

    def learn(self):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # === PERFORM OPTIMIZATION STEP === #
        primal_loss, dual_loss, infos = self.compute_loss()
        grad_dict = self.optimize_params(primal_loss, dual_loss)

        # === LOGGING === #
        supp_dict = {}
        if self.num_updates % 300 == 0:
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
            f"{self.name}/analytics/nu1": self.nu[0].item(),
            f"{self.name}/analytics/nu2": self.nu[1].item(),
            f"{self.name}/analytics/nu3": self.nu[2].item(),
            f"{self.name}/analytics/zeta": self.zeta.item(),
            f"{self.name}/lr/W_lr": self.lr_scheduler1.get_last_lr()[0],
            f"{self.name}/lr/u_lr": self.lr_scheduler1.get_last_lr()[1],
            f"{self.name}/lr/lbd_lr": self.lr_scheduler1.get_last_lr()[2],
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
