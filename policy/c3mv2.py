import time
from collections import deque
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inverse, matmul, transpose
from torch.autograd import grad
from torch.linalg import matrix_norm
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base


class C3Mv2(Base):
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
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 0,
        device: str = "cpu",
    ):
        super(C3Mv2, self).__init__()

        # constants
        self.name = "C3M"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = action_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size

        self.nupdates = nupdates
        self.current_update = 0

        # trainable networks
        self.W_func = W_func
        self.u_func = u_func

        self.data = data
        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            # set to eval mode due to dropout
            self.get_f_and_B.eval()

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

        self.eps = eps
        self.w_ub = w_ub

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": self.u_func.parameters(), "lr": u_lr},
                {"params": [self.lbd], "lr": 1e-3},
            ]
        )
        self.dual_optimizer = torch.optim.Adam(
            [{"params": [self.nu], "lr": 3e-3}, {"params": [self.zeta], "lr": 3e-3}]
        )

        self.lr_scheduler1 = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        self.lr_scheduler2 = LambdaLR(self.dual_optimizer, lr_lambda=self.lr_lambda)

        self.Cu_eigenvalues_records = []
        self.dot_M_eigenvalues_records = []
        self.sym_mabk_eigenvalues_records = []
        self.C1_eigenvalues_records = []
        self.C2_loss_records = []
        self.overshoot_records = []

        #
        self.num_W_update = 0
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

        x, xref, uref, t = self.trim_state(state)
        a, _ = self.u_func(x, xref, uref, deterministic=deterministic)

        return a, {
            "probs": self.dummy,  # dummy for code consistency
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

    # def learn(self):
    #     detach = True if self.num_W_update < int(0.05 * self.nupdates) else False
    #     loss_dict, supp_dict, update_time = self.learn_W(detach)
    #     self.num_W_update += 1

    #     return loss_dict, supp_dict, update_time

    def learn(self, detach: bool = False):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # first sample batch (size of 1024) from the data
        batch = dict()
        buffer_size, batch_size = self.data["x"].shape[0], 1024
        indices = np.random.choice(buffer_size, size=batch_size, replace=False)
        for key in self.data.keys():
            # Sample a batch of 1024
            batch[key] = self.data[key][indices]

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        #### COMPUTE INGREDIENTS ####
        x = to_tensor(batch["x"]).requires_grad_()
        xref = to_tensor(batch["xref"])
        uref = to_tensor(batch["uref"])

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
        dot_M = self.weighted_gradients(M, dot_x, x, detach)

        # contraction condition
        if detach:
            ABK = A + matmul(B, K)
            MABK = matmul(M.detach(), ABK)
            sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
            Cu = dot_M + sym_MABK + 2 * self.lbd * M.detach()
        else:
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

        #### COMPUTE LOSS ####
        pd_loss = self.loss_pos_matrix_random_sampling(-Cu)
        c1_loss = self.loss_pos_matrix_random_sampling(-C1)
        overshoot_loss = self.loss_pos_matrix_random_sampling(-overshoot)
        c2_loss = C2

        nu = self.nu.detach()
        zeta = self.zeta.detach()
        loss = (
            -self.lbd
            + nu[0] * overshoot_loss
            + nu[1] * pd_loss
            + nu[2] * c1_loss
            + zeta * c2_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func, self.u_func, self.lbd],
            ["W_func", "u_func", "lbd"],
            dir="C3M",
            device=self.device,
        )
        self.optimizer.step()

        dual_loss = -(
            self.nu[0] * overshoot_loss.detach()
            + self.nu[1] * pd_loss.detach()
            + self.nu[2] * c1_loss.detach()
            + self.zeta * c2_loss.detach()
        )
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

        # ensure lbd and nu are positive
        with torch.no_grad():
            self.lbd.clamp_(min=1e-6, max=1e6)
            self.nu.clamp_(min=0.0, max=1e6)

        with torch.no_grad():
            dot_M_eig = self.get_matrix_eig(dot_M)
            sym_MABK_eig = self.get_matrix_eig(sym_MABK)
            M_eig = self.get_matrix_eig(M)
            BK_eig = self.get_matrix_eig(matmul(B, K))
            overshoot_eig = self.get_matrix_eig(overshoot)

            Cu_eig = self.get_matrix_eig(Cu)
            C1_eig = self.get_matrix_eig(C1)

            self.Cu_eigenvalues_records.append(Cu_eig)
            self.dot_M_eigenvalues_records.append(dot_M_eig)
            self.sym_mabk_eigenvalues_records.append(sym_MABK_eig)
            self.C1_eigenvalues_records.append(C1_eig)
            self.C2_loss_records.append(C2.cpu().numpy())
            self.overshoot_records.append(overshoot_eig)

        ### LOGGING ###
        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/pd_loss": pd_loss.item(),
            f"{self.name}/loss/c1_loss": c1_loss.item(),
            f"{self.name}/loss/c2_loss": c2_loss.item(),
            f"{self.name}/loss/overshoot_loss": overshoot_loss.item(),
            f"{self.name}/analytics/lbd": self.lbd.item(),
            f"{self.name}/analytics/nu1": self.nu[0].item(),
            f"{self.name}/analytics/nu2": self.nu[1].item(),
            f"{self.name}/analytics/nu3": self.nu[2].item(),
            f"{self.name}/analytics/zeta": self.zeta.item(),
            f"{self.name}/analytics/M_eig_max": M_eig.max(),
            f"{self.name}/analytics/M_eig_min": M_eig.min(),
            f"{self.name}/analytics/BK_eig_max": BK_eig.max(),
            f"{self.name}/analytics/BK_eig_min": BK_eig.min(),
            f"{self.name}/lr/W_lr": self.lr_scheduler1.get_last_lr()[0],
            f"{self.name}/lr/u_lr": self.lr_scheduler1.get_last_lr()[1],
            f"{self.name}/lr/dual_lr": self.lr_scheduler2.get_last_lr()[0],
        }
        norm_dict = self.compute_weight_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        supp_dict = self.get_supp_dict()

        # Cleanup
        self.eval()
        self.current_update += 1

        update_time = time.time() - t0
        self.lr_scheduler1.step()
        self.lr_scheduler2.step()

        return loss_dict, supp_dict, update_time

    def get_supp_dict(self):
        ### DRAW THE FIGURE OF EIGENVALUES ###
        num = 10
        supp_dict = {}
        if (
            len(self.Cu_eigenvalues_records) >= num
            and len(self.C1_eigenvalues_records) >= num
        ):
            # make plt figure of cu and c1 eigenvalues
            x = list(range(0, len(self.Cu_eigenvalues_records), num))

            Cu_eig_array = np.asarray(self.Cu_eigenvalues_records[::num])  # every 10th
            dot_M_eig_array = np.asarray(
                self.dot_M_eigenvalues_records[::num]
            )  # every 10th
            sym_MABK_eig_array = np.asarray(
                self.sym_mabk_eigenvalues_records[::num]
            )  # every 10th
            C1_eig_array = np.asarray(self.C1_eigenvalues_records[::num])
            C2_loss = np.asarray(self.C2_loss_records[::num])
            overshoot_eig_array = np.asarray(self.overshoot_records[::num])

            # find mean and 95% confidence interval
            Cu_mean = Cu_eig_array.mean(axis=1)
            Cu_max = Cu_eig_array.max(axis=1)
            Cu_min = Cu_eig_array.min(axis=1)

            dot_M_mean = dot_M_eig_array.mean(axis=1)
            dot_M_max = dot_M_eig_array.max(axis=1)
            dot_M_min = dot_M_eig_array.min(axis=1)

            sym_MABK_mean = sym_MABK_eig_array.mean(axis=1)
            sym_MABK_max = sym_MABK_eig_array.max(axis=1)
            sym_MABK_min = sym_MABK_eig_array.min(axis=1)

            C1_mean = C1_eig_array.mean(axis=1)
            C1_max = C1_eig_array.max(axis=1)
            C1_min = C1_eig_array.min(axis=1)
            overshoot_mean = overshoot_eig_array.mean(axis=1)
            overshoot_max = overshoot_eig_array.max(axis=1)
            overshoot_min = overshoot_eig_array.min(axis=1)

            fig, ax = plt.subplots(2, 3, figsize=(12, 6))

            ax[0, 0].plot(
                x,
                Cu_mean,
                label=f"Cu Mean (max={Cu_max[-1]:.3g}, min={Cu_min[-1]:.3g})",
            )
            ax[0, 0].fill_between(x, Cu_max, Cu_min, alpha=0.2)
            ax[0, 0].set_title("Cu Eigenvalues")
            ax[0, 0].legend()

            ax[0, 1].plot(
                x,
                dot_M_mean,
                label=f"Dot M Mean (max={dot_M_max[-1]:.3g}, min={dot_M_min[-1]:.3g})",
            )
            ax[0, 1].fill_between(x, dot_M_max, dot_M_min, alpha=0.2)
            ax[0, 1].set_title("Dot M Eigenvalues")
            ax[0, 1].legend()

            ax[0, 2].plot(
                x,
                sym_MABK_mean,
                label=f"Sym MABK Mean (max={sym_MABK_max[-1]:.3g}, min={sym_MABK_min[-1]:.3g})",
            )
            ax[0, 2].fill_between(x, sym_MABK_max, sym_MABK_min, alpha=0.2)
            ax[0, 2].set_title("Sym MABK Eigenvalues")
            ax[0, 2].legend()

            ax[1, 0].plot(
                x,
                C1_mean,
                label=f"C1 Mean (max={C1_max[-1]:.3g}, min={C1_min[-1]:.3g})",
            )
            ax[1, 0].fill_between(x, C1_max, C1_min, alpha=0.2)
            ax[1, 0].set_title("C1 Eigenvalues")
            ax[1, 0].legend()

            supp_dict[f"{self.name}/analytics/C_eig"] = fig

            ax[1, 1].plot(
                x,
                C2_loss,
                label=f"C2 loss = {C2_loss[-1]:.3g}",
            )
            ax[1, 1].set_title("C2 Loss")
            # set y log scale
            ax[1, 1].set_yscale("log")
            ax[1, 1].legend()

            ax[1, 2].plot(
                x,
                overshoot_mean,
                label=f"Overshoot Mean (max={overshoot_max[-1]:.3g}, min={overshoot_min[-1]:.3g})",
            )
            ax[1, 2].fill_between(x, overshoot_max, overshoot_min, alpha=0.2)
            ax[1, 2].set_title("Overshoot Eigs")
            ax[1, 2].legend()

            ax[0, 0].grid(linestyle="--", alpha=0.5)
            ax[0, 1].grid(linestyle="--", alpha=0.5)
            ax[0, 2].grid(linestyle="--", alpha=0.5)
            ax[1, 0].grid(linestyle="--", alpha=0.5)
            ax[1, 1].grid(linestyle="--", alpha=0.5)
            ax[1, 1].grid(linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.close(fig)

        return supp_dict
