import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import inverse, matmul, transpose
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base


class C3M(Base):
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
        super(C3M, self).__init__()

        # constants
        self.name = "C3M"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = action_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size

        self.nupdates = nupdates

        # trainable networks
        self.W_func = W_func
        self.u_func = u_func

        self.data = data
        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            # set to eval mode due to dropout
            self.get_f_and_B.eval()

        self.eps = eps
        self.w_ub = w_ub
        self.w_lb = w_lb
        self.lbd = lbd
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": self.u_func.parameters(), "lr": u_lr},
            ]
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        self.num_updates = 0
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

    def compute_loss(self):
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
        Cu = dot_M + 2 * sym_MABK + 2 * self.lbd * M.detach()

        # C1
        DfW = self.weighted_gradients(W, f, x)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = 0.5 * (DfDxW + transpose(DfDxW, 1, 2))

        # this has to be a negative definite matrix
        C1_inner = -DfW + 2 * sym_DfDxW + 2 * self.lbd * W.detach()
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)

        C2_inners = []
        C2s = []
        for j in range(self.action_dim):
            DbW = self.weighted_gradients(W, B[:, :, j], x)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = DbDxW + transpose(DbDxW, 1, 2)
            C2_inner = DbW - 2 * sym_DbDxW
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

        loss = (
            overshoot_loss
            + pd_loss
            + c1_loss
            + c2_loss
            + pd_reg
            + c1_reg
            + overshoot_reg
        )

        return (
            loss,
            {
                "pd_loss": pd_loss,
                "c1_loss": c1_loss,
                "c2_loss": c2_loss,
                "overshoot_loss": overshoot_loss,
            },
        )

    def optimize_params(self, loss: torch.Tensor):
        # === OPTIMIZATION STEP === #
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

        # lr scheduling
        self.lr_scheduler.step()

        return grad_dict

    def learn(self):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # === PERFORM OPTIMIZATION STEP === #
        loss, infos = self.compute_loss()
        grad_dict = self.optimize_params(loss)

        # === LOGGING === #
        supp_dict = {}
        if self.num_updates % 500 == 0:
            fig = self.get_eigenvalue_plot()
            supp_dict["C3M/plot/eigenvalues"] = fig

        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/pd_loss": infos["pd_loss"].item(),
            f"{self.name}/loss/c1_loss": infos["c1_loss"].item(),
            f"{self.name}/loss/c2_loss": infos["c2_loss"].item(),
            f"{self.name}/loss/overshoot_loss": infos["overshoot_loss"].item(),
            f"{self.name}/lr/W_lr": self.lr_scheduler.get_last_lr()[0],
            f"{self.name}/lr/u_lr": self.lr_scheduler.get_last_lr()[1],
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
