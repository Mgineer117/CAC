import time
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inverse, matmul, transpose
from torch.linalg import matrix_norm
from torch.optim.lr_scheduler import LambdaLR

from policy.base import Base
from utils.functions import estimate_advantages


class CAC(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        W_func: nn.Module,
        get_f_and_B: Callable,
        actor: nn.Module,
        critic: nn.Module,
        data: dict,
        W_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        w_ub: float = 1e-2,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        eps_clip: float = 0.2,
        W_entropy_scaler: float = 1e-3,
        entropy_scaler: float = 1e-3,
        control_scaler: float = 0.0,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        nupdates: int = 1,
        dt: float = 0.03,
        device: str = "cpu",
    ):
        super(CAC, self).__init__()

        # constants
        self.name = "CAC"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = actor.action_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices

        self.data = data
        self.buffer_size = data["x"].shape[0]
        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self.W_entropy_scaler = W_entropy_scaler
        self.entropy_scaler = entropy_scaler
        self.control_scaler = control_scaler
        self.eps = eps
        self.gamma = gamma
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.target_kl = target_kl
        self.lbd = lbd
        self.eps_clip = eps_clip
        self.w_ub = w_ub
        self.dt = dt

        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            # set to eval mode due to dropout
            self.get_f_and_B.eval()

        self.nupdates = nupdates
        self.num_ppo_update = 0

        # trainable networks
        self.W_func = W_func
        self.W_optimizer = torch.optim.Adam(params=self.W_func.parameters(), lr=W_lr)

        self.actor = actor
        self.critic = critic

        self.ppo_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": actor_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        self.W_lr_scheduler = LambdaLR(self.W_optimizer, lr_lambda=self.W_lr_lambda)
        self.ppo_lr_scheduler = LambdaLR(
            self.ppo_optimizer, lr_lambda=self.ppo_lr_lambda
        )

        self.cmg_warmup = False

        self.to(self._dtype).to(self.device)

    def W_lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def ppo_lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        x, xref, uref = self.trim_state(state)
        a, metaData = self.actor(x, xref, uref, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def learn(self, batch):
        detach = True if self.num_ppo_update < int(0.1 * self.nupdates) else False

        loss_dict, update_time = {}, 0
        for _ in range(3):
            W_loss_dict, W_update_time = self.learn_W(batch, detach)
            update_time += W_update_time

        ppo_loss_dict, timesteps, ppo_update_time = self.learn_ppo(batch)

        loss_dict.update(W_loss_dict)
        loss_dict.update(ppo_loss_dict)
        update_time += ppo_update_time

        self.W_lr_scheduler.step()
        # self.ppo_lr_scheduler.step()

        self.num_ppo_update += 1

        return loss_dict, timesteps, update_time

    def learn_W(self, policy_batch: dict, detach: bool):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

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
        rewards = to_tensor(policy_batch["rewards"])

        if self.W_entropy_scaler == 0.0:
            W, infos = self.W_func(x, deterministic=True)
        else:
            W, infos = self.W_func(x)
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

        u = u.detach()
        K = K.detach()

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
            sym_MABK = MABK + transpose(MABK, 1, 2)
            C_u = dot_M + sym_MABK + 2 * self.lbd * M.detach()
        else:
            ABK = A + matmul(B, K)
            MABK = matmul(M, ABK)
            sym_MABK = MABK + transpose(MABK, 1, 2)
            C_u = dot_M + sym_MABK + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f, x)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = DfDxW + transpose(DfDxW, 1, 2)

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

        #### COMPUTE LOSS ####
        pd_loss = self.loss_pos_matrix_random_sampling(
            -C_u - self.eps * torch.eye(C_u.shape[-1]).to(self.device)
        )
        c1_loss = self.loss_pos_matrix_random_sampling(
            -C1 - self.eps * torch.eye(C1.shape[-1]).to(self.device)
        )
        c2_loss = sum([(C2**2).reshape(batch_size, -1).sum(1).mean() for C2 in C2s])
        # c2_loss = sum([(matrix_norm(C2) ** 2).mean() for C2 in C2s])
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            (self.w_ub * torch.eye(W.shape[-1])).unsqueeze(0).to(self.device) - W
        )

        ############# entropy loss ################
        mean_penalty = torch.exp(-rewards.mean())
        mean_entropy = infos["entropy"].mean()

        # prioritize the constraints than contraction
        # We use this as per our method to reduce
        # the complexity of bi-level optimization of RL
        weights = [0.1, 0.4, 0.4, 0.1]
        cmg_loss = (
            weights[0] * pd_loss
            + weights[1] * c1_loss
            + weights[2] * c2_loss
            + weights[3] * overshoot_loss
        )
        entropy_loss = self.W_entropy_scaler * mean_penalty * mean_entropy

        loss = cmg_loss  # - entropy_loss

        self.W_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.W_func.parameters(), max_norm=100.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func],
            ["W_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.W_optimizer.step()

        ### for loggings
        with torch.no_grad():
            dot_M_pos_eig, dot_M_neg_eig = self.get_matrix_eig(dot_M)
            sym_MABK_pos_eig, sym_MABK_neg_eig = self.get_matrix_eig(sym_MABK)
            M_pos_eig, M_neg_eig = self.get_matrix_eig(M)

            C_pos_eig, C_neg_eig = self.get_matrix_eig(C_u)
            C1_pos_eig, C1_neg_eig = self.get_matrix_eig(C1)

        # Logging
        loss_dict = {
            f"{self.name}/W_loss/loss": loss.item(),
            f"{self.name}/W_loss/pd_loss": pd_loss.item(),
            f"{self.name}/W_loss/c1_loss": c1_loss.item(),
            f"{self.name}/W_loss/c2_loss": c2_loss.item(),
            f"{self.name}/W_loss/overshoot_loss": overshoot_loss.item(),
            f"{self.name}/W_loss/entropy_loss": entropy_loss.item(),
            f"{self.name}/W_loss/mean_penalty": mean_penalty.item(),
            f"{self.name}/W_loss/mean_entropy": mean_entropy.item(),
            f"{self.name}/C_analytics/C_pos_eig": C_pos_eig.item(),
            f"{self.name}/C_analytics/C_neg_eig": C_neg_eig.item(),
            f"{self.name}/C_analytics/C1_pos_eig": C1_pos_eig.item(),
            f"{self.name}/C_analytics/C1_neg_eig": C1_neg_eig.item(),
            f"{self.name}/C_analytics/dot_M_pos_eig": dot_M_pos_eig.item(),
            f"{self.name}/C_analytics/dot_M_neg_eig": dot_M_neg_eig.item(),
            f"{self.name}/C_analytics/sym_MABK_pos_eig": sym_MABK_pos_eig.item(),
            f"{self.name}/C_analytics/sym_MABK_neg_eig": sym_MABK_neg_eig.item(),
            f"{self.name}/C_analytics/M_pos_eig": M_pos_eig.item(),
            f"{self.name}/C_analytics/M_neg_eig": M_neg_eig.item(),
            f"{self.name}/learning_rate/W_lr": self.W_optimizer.param_groups[0]["lr"],
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
        return loss_dict, update_time

    def learn_ppo(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])
        original_rewards = to_tensor(batch["rewards"])
        rewards = self.get_rewards(states, actions)
        terminals = to_tensor(batch["terminals"])
        old_logprobs = to_tensor(batch["logprobs"])

        # Compute advantages and returns
        with torch.no_grad():
            values = self.critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
                device=self.device,
            )

        # Mini-batch training
        batch_size = states.size(0)

        # List to track actor loss over minibatches
        losses = []
        actor_losses = []
        value_losses = []
        l2_losses = []
        entropy_losses = []

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_actions = states[indices], actions[indices]
                mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

                # advantages
                mb_advantages = advantages[indices]
                mb_advantages = (
                    mb_advantages - mb_advantages.mean()
                ) / mb_advantages.std()

                # 1. Critic Loss (with optional regularization)
                value_loss, l2_loss = self.critic_loss(mb_states, mb_returns)
                # Track value loss for logging
                value_losses.append(value_loss.item())
                l2_losses.append(l2_loss.item())

                # 2. actor Loss
                actor_loss, entropy_loss, clip_fraction, kl_div = self.actor_loss(
                    mb_states, mb_actions, mb_old_logprobs, mb_advantages
                )

                # Track actor loss for logging
                actor_losses.append(actor_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction)
                target_kl.append(kl_div.item())

                if kl_div.item() > self.target_kl:
                    break

                # Total loss
                loss = actor_loss - entropy_loss + 0.5 * value_loss + l2_loss
                losses.append(loss.item())

                # Update parameters
                self.ppo_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic],
                    ["actor", "critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.ppo_optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": np.mean(losses),
            f"{self.name}/loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/loss/value_loss": np.mean(value_losses),
            f"{self.name}/loss/l2_loss": np.mean(l2_losses),
            f"{self.name}/loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/analytics/klDivergence": target_kl[-1],
            f"{self.name}/analytics/K-epoch": k + 1,
            f"{self.name}/analytics/avg_rewards": torch.mean(original_rewards).item(),
            f"{self.name}/analytics/corrected_avg_rewards": torch.mean(rewards).item(),
            f"{self.name}/learning_rate/actor_lr": self.ppo_optimizer.param_groups[0][
                "lr"
            ],
            f"{self.name}/learning_rate/critic_lr": self.ppo_optimizer.param_groups[1][
                "lr"
            ],
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.actor, self.critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, actions, rewards, terminals, old_logprobs
        self.eval()

        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0
        return loss_dict, timesteps, update_time

    def actor_loss(
        self,
        mb_states: torch.Tensor,
        mb_actions: torch.Tensor,
        mb_old_logprobs: torch.Tensor,
        mb_advantages: torch.Tensor,
    ):
        x, xref, uref = self.trim_state(mb_states)

        _, metaData = self.actor(x, xref, uref)
        logprobs = self.actor.log_prob(metaData["dist"], mb_actions)
        entropy = self.actor.entropy(metaData["dist"])
        ratios = torch.exp(logprobs - mb_old_logprobs)

        surr1 = ratios * mb_advantages
        surr2 = (
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
        )

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = self.entropy_scaler * entropy.mean()

        # Compute clip fraction (for logging)
        clip_fraction = torch.mean(
            (torch.abs(ratios - 1) > self.eps_clip).float()
        ).item()

        # Check if KL divergence exceeds target KL for early stopping
        kl_div = torch.mean(mb_old_logprobs - logprobs)

        return actor_loss, entropy_loss, clip_fraction, kl_div

    def critic_loss(self, mb_states: torch.Tensor, mb_returns: torch.Tensor):
        mb_values = self.critic(mb_states)
        value_loss = self.mse_loss(mb_values, mb_returns)
        l2_loss = (
            sum(param.pow(2).sum() for param in self.critic.parameters()) * self.l2_reg
        )

        return value_loss, l2_loss
