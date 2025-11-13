import os
from abc import ABC, abstractmethod  # Added for abstract base class

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inverse, matmul, transpose
from torch.autograd import grad


# ===================================================================
# 1. UTILITIES CLASS (Grandparent)
# Handles plotting, logging, and device/tensor utilities
# ===================================================================
class Utilities(nn.Module):
    def __init__(self):
        super(Utilities, self).__init__()

        self._dtype = torch.float32
        self.device = None  # the algorithm should save its own device

        # Lists for logging eigenvalue data
        self.Cu_eigenvalues_records = []
        self.dot_M_eigenvalues_records = []
        self.sym_mabk_eigenvalues_records = []
        self.C1_eigenvalues_records = []
        self.C2_loss_records = []
        self.overshoot_records = []

        # Common loss functions
        self.l1_loss = F.l1_loss
        self.mse_loss = F.mse_loss
        self.huber_loss = F.smooth_l1_loss

    def to_tensor(self, data):
        """Converts numpy data to a tensor on the correct device."""
        return torch.from_numpy(data).to(self._dtype).to(self.device)

    def to_device(self, device):
        """Moves the entire module to the specified device."""
        self.device = device
        self.to(device)

    def get_matrix_eig(self, A: torch.Tensor):
        """Calculates and returns the mean eigenvalues of a symmetric matrix."""
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(A)  # (batch, dim), real symmetric
        return eigvals.mean(0).cpu().numpy()

    def average_dict_values(self, dict_list):
        """Averages values from a list of dictionaries."""
        if not dict_list:
            return {}

        sum_dict = {}
        count_dict = {}

        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}
        return avg_dict

    def compute_gradient_norm(self, models, names, device, dir="None", norm_type=2):
        """Computes the total L-norm of gradients for a list of models."""
        grad_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        if param.grad is not None:
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
        """Computes the total L-norm of weights for a list of models."""
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

    def record_eigenvalues(self, Cu, dot_M, sym_MABK, C1, C2, overshoot):
        """Records the eigenvalues of various matrices for logging."""
        with torch.no_grad():
            dot_M_eig = self.get_matrix_eig(dot_M)
            sym_MABK_eig = self.get_matrix_eig(sym_MABK)
            overshoot_eig = self.get_matrix_eig(overshoot)
            Cu_eig = self.get_matrix_eig(Cu)
            C1_eig = self.get_matrix_eig(C1)

            self.Cu_eigenvalues_records.append(Cu_eig)
            self.dot_M_eigenvalues_records.append(dot_M_eig)
            self.sym_mabk_eigenvalues_records.append(sym_MABK_eig)
            self.C1_eigenvalues_records.append(C1_eig)
            self.C2_loss_records.append(C2.cpu().numpy())
            self.overshoot_records.append(overshoot_eig)

    def get_eigenvalue_plot(self):
        """Generates a Matplotlib figure of the recorded eigenvalues."""
        num = 10
        if (
            len(self.Cu_eigenvalues_records) < num
            or len(self.C1_eigenvalues_records) < num
        ):
            return None  # Not enough data to plot

        x = list(range(0, len(self.Cu_eigenvalues_records), num))

        Cu_eig_array = np.asarray(self.Cu_eigenvalues_records[::num])
        dot_M_eig_array = np.asarray(self.dot_M_eigenvalues_records[::num])
        sym_MABK_eig_array = np.asarray(self.sym_mabk_eigenvalues_records[::num])
        C1_eig_array = np.asarray(self.C1_eigenvalues_records[::num])
        C2_loss = np.asarray(self.C2_loss_records[::num])
        overshoot_eig_array = np.asarray(self.overshoot_records[::num])

        Cu_mean, Cu_max, Cu_min = (
            Cu_eig_array.mean(axis=1),
            Cu_eig_array.max(axis=1),
            Cu_eig_array.min(axis=1),
        )
        dot_M_mean, dot_M_max, dot_M_min = (
            dot_M_eig_array.mean(axis=1),
            dot_M_eig_array.max(axis=1),
            dot_M_eig_array.min(axis=1),
        )
        sym_MABK_mean, sym_MABK_max, sym_MABK_min = (
            sym_MABK_eig_array.mean(axis=1),
            sym_MABK_eig_array.max(axis=1),
            sym_MABK_eig_array.min(axis=1),
        )
        C1_mean, C1_max, C1_min = (
            C1_eig_array.mean(axis=1),
            C1_eig_array.max(axis=1),
            C1_eig_array.min(axis=1),
        )
        overshoot_mean, overshoot_max, overshoot_min = (
            overshoot_eig_array.mean(axis=1),
            overshoot_eig_array.max(axis=1),
            overshoot_eig_array.min(axis=1),
        )

        fig, ax = plt.subplots(2, 3, figsize=(12, 6))

        ax[0, 0].plot(
            x, Cu_mean, label=f"Cu Mean (max={Cu_max[-1]:.3g}, min={Cu_min[-1]:.3g})"
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
            x, C1_mean, label=f"C1 Mean (max={C1_max[-1]:.3g}, min={C1_min[-1]:.3g})"
        )
        ax[1, 0].fill_between(x, C1_max, C1_min, alpha=0.2)
        ax[1, 0].set_title("C1 Eigenvalues")
        ax[1, 0].legend()

        ax[1, 1].plot(x, C2_loss, label=f"C2 loss = {C2_loss[-1]:.3g}")
        ax[1, 1].set_title("C2 Loss")
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

        for i in range(2):
            for j in range(3):
                ax[i, j].grid(linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.close(fig)  # Close the figure to prevent display issues

        return fig


# ===================================================================
# 2. BASE CLASS (Parent)
# Inherits from Utilities
# Handles advanced autograd/math (Jacobians, B_perp, etc.)
# ===================================================================
class Base(Utilities, ABC):  # Inherit from Utilities and make abstract
    def __init__(self):
        # Initialize the parent class (Utilities)
        # This gives Base all the methods and attributes from Utilities
        super(Base, self).__init__()

        # Note: All the lists (Cu_eigenvalues_records, etc.)
        # and loss functions (l1_loss, etc.) are
        # automatically inherited. No need to redefine them.

    def trim_state(self, state: torch.Tensor):
        """Trims a state tensor into its components (x, xref, uref, t)."""
        # state trimming
        x = state[:, : self.x_dim].requires_grad_()
        xref = state[:, self.x_dim : 2 * self.x_dim].requires_grad_()
        uref = state[
            :, 2 * self.x_dim : 2 * self.x_dim + self.action_dim
        ].requires_grad_()
        t = state[:, -1].unsqueeze(-1)

        return x, xref, uref, t

    def extract_trajectories(self, x: torch.Tensor, terminals: torch.Tensor) -> list:
        """Extracts individual trajectories from a batch based on terminal flags."""
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

        # If there are remaining states not ended by a terminal flag, add them.
        if len(x_list) > 0:
            traj_x_list.append(torch.stack(x_list, dim=0))

        return traj_x_list

    def compute_B_perp_batch(self, B, B_perp_dim):
        """
        Compute a batch of B_perp matrices (orthogonal complement) in parallel.
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

    def loss_pos_matrix_random_sampling(self, A: torch.Tensor, reg: bool = True):
        """
        Calculates a loss for non-positive-definite matrices
        using random sampling. A is (n, d, d).
        """
        n, A_dim, _ = A.shape

        z = torch.randn((n, A_dim)).to(dtype=self._dtype, device=self.device)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.unsqueeze(-1)
        zT = transpose(z, 1, 2)

        zTAz = matmul(matmul(zT, A), z)

        loss_eigen = torch.relu(-zTAz).mean()
        loss_reg = torch.relu(zTAz - 50).mean()

        return loss_eigen, loss_reg if reg else 0

    def Jacobian(self, f: torch.Tensor, x: torch.Tensor):
        """Computes the Jacobian of a vector f w.r.t. vector x."""
        # NOTE that this function assume that data are independent of each other
        f = f + 0.0 * x.sum()  # to avoid the case that f is independent of x

        n = x.shape[0]
        f_dim = f.shape[-1]
        x_dim = x.shape[-1]

        J = torch.zeros(n, f_dim, x_dim).to(dtype=self._dtype, device=self.device)
        for i in range(f_dim):
            J[:, i, :] = grad(f[:, i].sum(), x, create_graph=True)[0]
        return J

    def Jacobian_Matrix(self, M: torch.Tensor, x: torch.Tensor):
        """Computes the Jacobian of a matrix M w.r.t. vector x."""
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
        """ComputOverwrites (if exists) or creates a file with the given content. This is a helper function for other tools and isn't intended to be used directly by the user."""
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
        """Computes weighted gradients of a matrix W."""
        assert v.size() == x.size()

        bs = x.shape[0]
        if detach:
            return (self.Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(
                dim=3
            )
        else:
            return (self.Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

    # This method must be implemented by any child class
    @abstractmethod
    def learn(self):
        """The main training loop for the algorithm."""
        pass


# ===================================================================
# 3. EXAMPLE ALGORITHM (Child)
# Inherits from Base
# This is where you would define your networks and 'learn' method
# ===================================================================


class MyAlgorithm(Base):
    def __init__(self, x_dim, action_dim, control_scaler=0.1):
        # Initialize the parent class (Base)
        super(MyAlgorithm, self).__init__()

        # --- Define Algorithm-Specific Properties ---
        self.x_dim = x_dim
        self.action_dim = action_dim
        self.control_scaler = control_scaler

        # --- Define Neural Networks ---
        # Example: A simple network for the W_func needed by get_rewards
        # You would replace this with your actual network definitions
        self.W_func = nn.Sequential(
            nn.Linear(self.x_dim, 64),
            nn.ReLU(),
            # ... etc ...
            # This is just a placeholder
            nn.Linear(64, self.x_dim * self.x_dim),
        ).to(self.device)

        # ... (Define your other networks: policy, critic, etc.) ...

    @abstractmethod
    def learn(self, data_batch):
        """
        Implement the main training logic here.
        This method is called to train the algorithm.
        """
        print("Learning from batch...")
        # 1. Get data from batch
        # state, action, reward, next_state = data_batch

        # 2. Use helper functions inherited from Base
        # e.g., jacobian = self.Jacobian(f_x, x)

        # 3. Calculate losses
        # e.g., loss = self.mse_loss(a, b)

        # 4. Backpropagate and update
        # ...

        # 5. Log eigenvalues (inherited from Utilities)
        # self.record_eigenvalues(...)

        pass


# Example of how to use it:
#
# my_algo = MyAlgorithm(x_dim=10, action_dim=4)
# my_algo.to_device(torch.device("cuda"))
#
# for batch in data_loader:
#     my_algo.learn(batch)
#
# fig = my_algo.get_eigenvalue_plot()
# if fig:
#     fig.savefig("my_plot.png")
