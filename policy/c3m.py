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
        super(C3M, self).__init__()

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
        self.cmg_warmup = False
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
        a, _ = self.u_func(x, xref, uref)

        return a, {
            "probs": self.dummy,  # dummy for code consistency
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

    def learn(self):
        detach = True if self.num_W_update < int(0.1 * self.nupdates) else False
        loss_dict, update_time = self.learn_W(detach)

        self.num_W_update += 1

        return loss_dict, update_time

    def learn_W(self, detach: bool):
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

        #  DBDx[:, :, :, i]: n, x_dim, x_dim
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
            C_u = dot_M + 2 * sym_MABK + 2 * self.lbd * M.detach()
        else:
            ABK = A + matmul(B, K)
            MABK = matmul(M, ABK)
            sym_MABK = 0.5 * (MABK + transpose(MABK, 1, 2))
            C_u = dot_M + 2 * sym_MABK + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f, x)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = 0.5 * (DfDxW + transpose(DfDxW, 1, 2))

        # this has to be a negative definite matrix
        C1_inner = -DfW + 2 * sym_DfDxW + 2 * self.lbd * W
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

        #### COMPUTE LOSS ####
        pd_loss = self.loss_pos_matrix_random_sampling(
            -C_u - self.eps * torch.eye(C_u.shape[-1]).to(self.device)
        )
        c1_loss = self.loss_pos_matrix_random_sampling(
            -C1 - self.eps * torch.eye(C1.shape[-1]).to(self.device)
        )
        c2_loss = sum([(C2**2).reshape(batch_size, -1).sum(1).mean() for C2 in C2s])
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            (self.w_ub * torch.eye(W.shape[-1])).unsqueeze(0).to(self.device) - W
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
        self.eval()
        self.current_update += 1

        update_time = time.time() - t0
        self.lr_scheduler.step()

        return loss_dict, update_time
