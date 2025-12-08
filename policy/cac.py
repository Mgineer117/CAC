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
from utils.functions import (
    compute_kl,
    conjugate_gradients,
    estimate_advantages,
    flat_params,
    hessian_vector_product,
    set_flat_params,
)


class CAC(Base):
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
        reward_mode: str = "default",
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
        super(CAC, self).__init__()

        # constants
        self.name = "CAC"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = actor.action_dim

        self.data = data
        self.buffer_size = data["x"].shape[0]
        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self.W_entropy_scaler = W_entropy_scaler
        self.entropy_scaler = entropy_scaler
        self.tracking_scaler = tracking_scaler
        self.control_scaler = control_scaler
        self.eps = eps
        self.gamma = gamma
        self.reward_mode = reward_mode
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.lbd = lbd
        self.damping = damping
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.target_kl = target_kl
        self.eps_clip = eps_clip
        self.w_ub = w_ub
        self.w_lb = w_lb

        self.get_f_and_B = get_f_and_B
        if isinstance(self.get_f_and_B, nn.Module):
            # set to eval mode due to dropout
            self.get_f_and_B.eval()

        self.nupdates = nupdates
        self.num_W_updates = 0
        self.num_RL_updates = 0

        # trainable networks
        self.W_func = W_func
        self.actor = actor
        self.critic = critic

        self.W_optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
            ]
        )

        self.RL_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": actor_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        self.progress = 0.0
        self.lr_scheduler1 = LambdaLR(
            self.W_optimizer, lr_lambda=self.timestep_lr_lambda
        )

        self.to(self._dtype).to(self.device)

    def timestep_lr_lambda(self, _):
        """
        Calculates LR multiplier based on total environment steps taken.
        Ignores the internal scheduler 'step' counter (_).
        Squared decay: drops faster than linear.
        """
        # We square the linear term.
        # Logic: At progress 0.5, linear is 0.5, quadratic is 0.25 (lower).
        return max(0.0, 1.0 - self.progress) ** 2

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        x, xref, uref, t = self.trim_state(state)
        a, metaData = self.actor(x, xref, uref, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

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
            sym_DbDxW = 0.5 * (DbDxW + transpose(DbDxW, 1, 2))
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

        if self.progress < 0.1:
            loss = c1_loss + c2_loss + c1_reg + overshoot_loss + overshoot_reg
        else:
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

    def optimize_W_params(self, loss: torch.Tensor):
        # === OPTIMIZATION STEP === #
        self.W_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func, self.lbd],
            ["W_func", "lbd"],
            dir="CAC",
            device=self.device,
        )
        self.W_optimizer.step()

        return grad_dict

    def learn(self, batch: dict, progress: float):
        self.progress = progress

        loss_dict, supp_dict = {}, {}

        # Implement the freeze-and-learn scheme here
        W_update_time = 0
        if self.num_RL_updates % 5 == 0:
            W_loss_dict, W_supp_dict, W_update_time = self.learn_W()
            loss_dict.update(W_loss_dict)
            supp_dict.update(W_supp_dict)

        RL_loss_dict, RL_supp_dict, RL_update_time = self.learn_ppo(batch)
        # RL_loss_dict, RL_supp_dict, RL_update_time = self.learn_trpo(batch)

        loss_dict.update(RL_loss_dict)
        supp_dict.update(RL_supp_dict)

        self.lr_scheduler1.step()

        update_time = W_update_time + RL_update_time
        self.num_RL_updates += 1

        return loss_dict, supp_dict, update_time

    def learn_W(self):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # === PERFORM OPTIMIZATION STEP === #
        loss, infos = self.compute_W_loss()
        grad_dict = self.optimize_W_params(loss)

        # === LOGGING === #
        supp_dict = {}
        if self.num_W_updates % 300 == 0:
            fig = self.get_eigenvalue_plot()
            supp_dict["CAC/plot/eigenvalues"] = fig

        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/pd_loss": infos["pd_loss"].item(),
            f"{self.name}/loss/c1_loss": infos["c1_loss"].item(),
            f"{self.name}/loss/c2_loss": infos["c2_loss"].item(),
            f"{self.name}/loss/overshoot_loss": infos["overshoot_loss"].item(),
            f"{self.name}/analytics/lbd": self.lbd,
            f"{self.name}/lr/W_lr": self.lr_scheduler1.get_last_lr()[0],
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

    def learn_ppo(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        states = self.to_tensor(batch["states"])
        actions = self.to_tensor(batch["actions"])
        original_rewards = self.to_tensor(batch["rewards"])
        rewards = self.get_rewards(states, actions)
        terminals = self.to_tensor(batch["terminals"])
        old_logprobs = self.to_tensor(batch["logprobs"])

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
        losses, actor_losses, value_losses, entropy_losses = [], [], [], []
        clip_fractions, target_kl, grad_dicts = [], [], []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_actions = states[indices], actions[indices]
                mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

                # advantages
                mb_advantages = advantages[indices]

                # 1. Critic Loss (with optional regularization)
                value_loss = self.critic_loss(mb_states, mb_returns)
                critic_loss = 0.5 * value_loss

                # Track value loss for logging
                value_losses.append(value_loss.item())

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
                loss = actor_loss - entropy_loss + 0.5 * critic_loss
                losses.append(loss.item())

                # Update parameters
                self.RL_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic],
                    ["actor", "critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.RL_optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        # Logging
        supp_dict = {}
        loss_dict = {
            f"{self.name}/loss/loss": np.mean(losses),
            f"{self.name}/loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/loss/value_loss": np.mean(value_losses),
            f"{self.name}/loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/analytics/klDivergence": target_kl[-1],
            f"{self.name}/analytics/K-epoch": k + 1,
            f"{self.name}/analytics/avg_rewards": torch.mean(original_rewards).item(),
            f"{self.name}/analytics/corrected_avg_rewards": torch.mean(rewards).item(),
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

        update_time = time.time() - t0
        return loss_dict, supp_dict, update_time

    def learn_trpo(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        states = self.to_tensor(batch["states"])
        actions = self.to_tensor(batch["actions"])
        original_rewards = self.to_tensor(batch["rewards"])
        rewards = self.get_rewards(states, actions)
        terminals = self.to_tensor(batch["terminals"])
        old_logprobs = self.to_tensor(batch["logprobs"])

        x, xref, uref, t = self.trim_state(states)

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

        # === CRITIC UPDATE === #
        batch_size = states.size(0) // self.num_minibatch
        grad_dict_list = []
        for _ in range(self.K):
            for _ in range(self.num_minibatch):
                indices = torch.randperm(states.size(0))[:batch_size]
                mb_states, mb_returns = states[indices], returns[indices]

                value_loss = self.critic_loss(mb_states, mb_returns)

                self.RL_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.critic],
                    ["critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dict_list.append(grad_dict)
                self.RL_optimizer.step()
        grad_dict = self.average_dict_values(grad_dict_list)

        # === ACTOR UPDATE === #
        actor_loss, entropy_loss, _, _ = self.actor_loss(
            states, actions, old_logprobs, advantages
        )
        loss = actor_loss + entropy_loss
        actor_gradients = torch.autograd.grad(loss, self.actor.parameters())
        grad_flat = torch.cat([g.view(-1) for g in actor_gradients]).detach()

        old_actor = deepcopy(self.actor)

        # KL function (closure)
        def kl_fn():
            return compute_kl(old_actor, self.actor, (x, xref, uref))

        # Define HVP function
        Hv = lambda v: hessian_vector_product(kl_fn, self.actor, self.damping, v)

        # Compute step direction with CG
        step_dir = conjugate_gradients(Hv, grad_flat, nsteps=10)

        # Compute step size to satisfy KL constraint
        sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
        lm = torch.sqrt(sAs / self.target_kl)
        full_step = step_dir / (lm + 1e-8)

        # Line search
        with torch.no_grad():
            old_params = flat_params(self.actor)

            # Backtracking line search
            success = False
            for i in range(self.backtrack_iters):
                step_frac = self.backtrack_coeff**i
                new_params = old_params - step_frac * full_step
                set_flat_params(self.actor, new_params)
                kl = compute_kl(old_actor, self.actor, (x, xref, uref))

                if kl <= self.target_kl:
                    success = True
                    break

            if not success:
                set_flat_params(self.actor, old_params)

        # Logging
        supp_dict = {}
        loss_dict = {
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/value_loss": value_loss.item(),
            f"{self.name}/loss/entropy_loss": entropy_loss.item(),
            f"{self.name}/analytics/backtrack_iter": i,
            f"{self.name}/analytics/backtrack_success": int(success),
            f"{self.name}/analytics/klDivergence": kl.item(),
            f"{self.name}/analytics/avg_rewards": torch.mean(original_rewards).item(),
            f"{self.name}/analytics/corrected_avg_rewards": torch.mean(rewards).item(),
            f"{self.name}/grad/actor": torch.linalg.norm(grad_flat).item(),
            f"{self.name}/analytics/step_norm": torch.linalg.norm(
                step_frac * full_step
            ).item(),
        }
        norm_dict = self.compute_weight_norm(
            [self.actor, self.critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(norm_dict)
        loss_dict.update(grad_dict)

        # Cleanup
        del states, actions, rewards, terminals, old_logprobs
        self.eval()
        update_time = time.time() - t0

        return loss_dict, supp_dict, update_time

    def get_rewards(self, states: torch.Tensor, actions: torch.Tensor):
        x, xref, uref, t = self.trim_state(states)

        tracking_error = (x - xref).unsqueeze(-1)
        control_effort = torch.linalg.norm(actions, dim=-1, keepdim=True)

        with torch.no_grad():
            ### Compute the main rewards
            W, _ = self.W_func(x, deterministic=True)
            # Add lower-bound scaled identity to guarantee positive definiteness
            W += self.w_lb * torch.eye(self.x_dim).to(self.device).view(
                1, self.x_dim, self.x_dim
            )
            M = torch.inverse(W)

            tracking_errorT = transpose(tracking_error, 1, 2)

            tracking_reward = -self.tracking_scaler * (
                tracking_errorT @ M @ tracking_error
            ).squeeze(-1)

            control_reward = -self.control_scaler * control_effort

        if self.reward_mode == "inverse":
            tracking_reward = 1 / (1 + abs(tracking_reward))
            control_reward = 1 / (1 + abs(control_reward))

        rewards = (0.5 * tracking_reward) + (0.5 * control_reward)

        return rewards

    def actor_loss(
        self,
        mb_states: torch.Tensor,
        mb_actions: torch.Tensor,
        mb_old_logprobs: torch.Tensor,
        mb_advantages: torch.Tensor,
    ):
        # Check if *any* element in the tensor is NaN
        x, xref, uref, t = self.trim_state(mb_states)

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
        return value_loss
