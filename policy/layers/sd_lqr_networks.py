import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inverse, matmul, transpose
from torch.autograd import grad
from torch.optim.lr_scheduler import LambdaLR

from policy.layers.building_blocks import MLP


class SDCLearner(nn.Module):
    def __init__(
        self,
        x_dim: int,
        a_dim: int,
        hidden_dim: list,
        get_f_and_B: Callable,
        nupdates: int,
        drop_out: float = 0.0,
        activation: nn.Module = nn.Tanh(),
        sdc_lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        super(SDCLearner, self).__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.model = MLP(
            2 * x_dim,
            hidden_dim,
            (a_dim + 1) * x_dim**2,
            activation=activation,
            dropout_rate=drop_out,
        )

        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            # set to eval mode due to dropout
            self.get_f_and_B.eval()
        self.nupdates = nupdates
        self.sdc_lr = sdc_lr

        self.SDC_optimizer = torch.optim.Adam(params=self.parameters(), lr=sdc_lr)
        self.SDC_lr_scheduler = LambdaLR(self.SDC_optimizer, lr_lambda=self.lr_lambda)

        self.name = "SDCLearner"
        self.device = device
        self.to(self.device)

    def lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def to_tensor(self, data):
        return torch.from_numpy(data).to(self.device)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            x = self.to_tensor(x)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        logits = self.model(x)

        # Split the logits into two parts
        Af = logits[:, : self.x_dim**2].reshape(-1, self.x_dim, self.x_dim)
        Bf = logits[:, self.x_dim**2 :].reshape(-1, self.a_dim, self.x_dim, self.x_dim)

        return Af, Bf

    def learn(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        xref = self.to_tensor(batch["xref"])
        uref = self.to_tensor(batch["uref"])

        x = self.to_tensor(batch["x"])
        u = uref + torch.randn_like(uref) * 0.1

        ####### LEARN SDC MODEL #######
        e = x - xref
        v = u - uref
        with torch.no_grad():
            f_x, B_x, _ = self.get_f_and_B(x)
            f_xref, B_xref, _ = self.get_f_and_B(xref)

            dot_e = (
                f_x
                + matmul(B_x, u.unsqueeze(-1)).squeeze(-1)
                - f_xref
                - matmul(B_xref, uref.unsqueeze(-1)).squeeze(-1)
            )

        sdc_input = torch.concatenate((x, e), dim=-1)
        Af, Bf = self(sdc_input)

        Af_e = matmul(Af, e.unsqueeze(-1)).squeeze(-1)

        Bf_e = matmul(Bf, e.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
        Bf_u = matmul(uref.unsqueeze(1), Bf_e).squeeze()

        dot_e_approx = Af_e + Bf_u + matmul(B_x, v.unsqueeze(-1)).squeeze(-1)

        sdc_loss = F.mse_loss(dot_e, dot_e_approx)

        # auxiliary loss
        f_diff = f_x - f_xref
        B_diff = transpose(B_x - B_xref, -1, -2)

        aux_loss = F.mse_loss(f_diff, Af_e) + F.mse_loss(B_diff, Bf_e)

        loss = sdc_loss + aux_loss

        self.SDC_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.SDC_optimizer.step()
        self.SDC_lr_scheduler.step()

        ####### FOR LOGGING #######
        loss_dict = {
            f"{self.name}/loss/sdc_loss": sdc_loss.item(),
            f"{self.name}/loss/aux_loss": aux_loss.item(),
            f"{self.name}/analytics/SDC_lr": self.SDC_optimizer.param_groups[0]["lr"],
        }

        update_time = time.time() - t0

        return loss_dict, update_time
