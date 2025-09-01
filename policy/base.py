import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inverse, matmul, transpose
from torch.autograd import grad


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

        self._dtype = torch.float32
        self.device = torch.device("cpu")

        # utils
        self.l1_loss = F.l1_loss
        self.mse_loss = F.mse_loss
        self.huber_loss = F.smooth_l1_loss

    def to_tensor(self, data):
        return torch.from_numpy(data).to(self._dtype).to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def extract_trajectories(self, x: torch.Tensor, terminals: torch.Tensor) -> list:
        traj_x_list = []
        x_list = []

        terminals = terminals.squeeze().tolist()

        for i in range(x.shape[0]):
            x_list.append(x[i])
            if terminals[i]:
                # Terminal state encountered: finalize current trajectory.
                x_tensor = torch.stack(x_list, dim=0)
                traj_x_list.append(x_tensor)
                x_list = []

        # If there are remaining states not ended by a terminal flag, add them as well.
        if len(x_list) > 0:
            traj_x_list.append(torch.stack(x_list, dim=0))

        return traj_x_list

    def compute_B_perp_batch(self, B, B_perp_dim):
        """
        Compute a batch of B_perp matrices in parallel.
        """
        batch_size, x_dim, _ = B.shape

        # Perform batched SVD
        U, S, Vh = torch.linalg.svd(B)  # U: (batch, x_dim, x_dim)

        # For each batch element, select columns beyond the rank
        B_perp = []
        for i in range(batch_size):
            U_i = U[i]  # (x_dim, x_dim)
            B_perp_i = U_i[:, -B_perp_dim:]  # (x_dim, x_dim - rank_i)

            # Pad or truncate to fixed B_perp_dim
            padded = torch.zeros(x_dim, B_perp_dim, device=B.device, dtype=B.dtype)
            m = B_perp_i.shape[1]
            if m > 0:
                padded[:, : min(m, B_perp_dim)] = B_perp_i[:, :B_perp_dim]
            B_perp.append(padded)

        # Stack
        B_perp_tensor = torch.stack(B_perp, dim=0)  # (batch, x_dim, B_perp_dim)

        return B_perp_tensor

    def loss_pos_matrix_random_sampling(self, A: torch.Tensor):
        # A: n x d x d
        # z: K x d
        n, A_dim, _ = A.shape

        z = torch.randn((n, A_dim)).to(dtype=self._dtype, device=self.device)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.unsqueeze(-1)
        zT = transpose(z, 1, 2)

        # K x d @ d x d = n x K x d
        zTAz = matmul(matmul(zT, A), z)

        negative_index = zTAz.detach().cpu().numpy() < 0
        if negative_index.sum() > 0:
            negative_zTAz = zTAz[negative_index]
            return -1.0 * (negative_zTAz.mean())
        else:
            return (
                torch.tensor(0.0)
                .to(dtype=self._dtype, device=self.device)
                .requires_grad_()
            )

    def rewards_pos_matrix_random_sampling(self, A: torch.Tensor):
        # A: n x d x d
        # z: K x d
        n, A_dim, _ = A.shape

        z = torch.randn((n, A_dim)).to(dtype=self._dtype, device=self.device)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.unsqueeze(-1)
        zT = transpose(z, 1, 2)

        # K x d @ d x d = n x K x d
        zTAz = matmul(matmul(zT, A), z)
        return zTAz.squeeze(-1)

    def trim_state(self, state: torch.Tensor):
        # state trimming
        x = state[:, : self.x_dim].requires_grad_()
        xref = state[:, self.x_dim : 2 * self.x_dim].requires_grad_()
        uref = state[
            :, 2 * self.x_dim : 2 * self.x_dim + self.action_dim
        ].requires_grad_()
        t = state[:, -1].unsqueeze(-1)

        return x, xref, uref, t

    def get_matrix_eig(self, A: torch.Tensor):
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(A)  # (batch, dim), real symmetric
            pos_eigvals = torch.relu(eigvals)
            neg_eigvals = torch.relu(-eigvals)
        return pos_eigvals.mean(dim=1).mean(), neg_eigvals.mean(dim=1).mean()

    def Jacobian(self, f: torch.Tensor, x: torch.Tensor):
        # NOTE that this function assume that data are independent of each other
        f = f + 0.0 * x.sum()  # to avoid the case that f is independent of x

        n = x.shape[0]
        f_dim = f.shape[-1]
        x_dim = x.shape[-1]

        J = torch.zeros(n, f_dim, x_dim).to(
            dtype=self._dtype, device=self.device
        )  # .to(x.type())
        for i in range(f_dim):
            J[:, i, :] = grad(f[:, i].sum(), x, create_graph=True)[0]  # [0]
        return J

    def Jacobian_Matrix(self, M: torch.Tensor, x: torch.Tensor):
        # NOTE that this function assume that data are independent of each other
        # M = M + 0.0 * x.sum()  # to avoid the case that f is independent of x

        n = x.shape[0]
        matrix_dim = M.shape[-1]
        x_dim = x.shape[-1]

        J = torch.zeros(n, matrix_dim, matrix_dim, x_dim).to(
            dtype=self._dtype, device=self.device
        )
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0]

        return J

    def B_Jacobian(self, B: torch.Tensor, x: torch.Tensor):
        n = x.shape[0]
        x_dim = x.shape[-1]

        DBDx = torch.zeros(n, x_dim, x_dim, self.action_dim).to(
            dtype=self._dtype, device=self.device
        )
        for i in range(self.action_dim):
            DBDx[:, :, :, i] = self.Jacobian(B[:, :, i].unsqueeze(-1), x)
        return DBDx

    def weighted_gradients(
        self, W: torch.Tensor, v: torch.Tensor, x: torch.Tensor, detach: bool = False
    ):
        # v, x: bs x n x 1
        # DWDx: bs x n x n x n
        assert v.size() == x.size()

        bs = x.shape[0]
        if detach:
            return (self.Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(
                dim=3
            )
        else:
            return (self.Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}

        return avg_dict

    def compute_gradient_norm(self, models, names, device, dir="None", norm_type=2):
        grad_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        if (
                            param.grad is not None
                        ):  # Only consider parameters that have gradients
                            param_grad_norm = torch.norm(param.grad, p=norm_type)
                            total_norm += param_grad_norm**norm_type
                except:
                    try:
                        param_grad_norm = torch.norm(model.grad, p=norm_type)
                    except:
                        param_grad_norm = torch.tensor(0.0)
                    total_norm += param_grad_norm**norm_type

                total_norm = total_norm ** (1.0 / norm_type)
                grad_dict[dir + "/grad/" + names[i]] = total_norm.item()

        return grad_dict

    def compute_weight_norm(self, models, names, device, dir="None", norm_type=2):
        norm_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        param_norm = torch.norm(param, p=norm_type)
                        total_norm += param_norm**norm_type
                except:
                    param_norm = torch.norm(model, p=norm_type)
                    total_norm += param_norm**norm_type
                total_norm = total_norm ** (1.0 / norm_type)
                norm_dict[dir + "/weight/" + names[i]] = total_norm.item()

        return norm_dict

    def get_rewards(self, states: torch.Tensor, actions: torch.Tensor):
        x, xref, uref, t = self.trim_state(states)
        with torch.no_grad():
            ### Compute the main rewards
            W, _ = self.W_func(x, deterministic=True)
            M = torch.inverse(W)

            error = (x - xref).unsqueeze(-1)
            errorT = transpose(error, 1, 2)

            rewards = (1 / (errorT @ M @ error + 1)).squeeze(-1)

        ### Compute the aux rewards ###
        fuel_efficiency = 1 / (torch.linalg.norm(actions, dim=-1, keepdim=True) + 1)
        rewards = rewards + self.control_scaler * fuel_efficiency

        return rewards

    def learn(self):
        pass
