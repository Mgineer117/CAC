from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from tqdm import tqdm

# PVTOL PARAMETERS
p_lim = np.pi / 3
pd_lim = np.pi / 3
vx_lim = 2.0
vz_lim = 1.0

X_MIN = np.array([-15.0, 0.0, -p_lim, -vx_lim, -vz_lim, -pd_lim]).reshape(-1, 1)
X_MAX = np.array([15.0, 30.0, p_lim, vx_lim, vz_lim, pd_lim]).reshape(-1, 1)

m = 0.486
J = 0.00383
g = 9.81
l = 0.25

lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim]).reshape(-1, 1)

# for sampling reference trajectory
X_INIT_MIN = np.array([-1, 1.0, -0.1, 0.5, 0.0, 0.0])
X_INIT_MAX = np.array([1, 2.0, 0.1, 1.0, 0.0, 0.0])

XE_INIT_MIN = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
XE_INIT_MAX = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

UREF_MIN = np.array([m * g / 2 - 1, m * g / 2 - 1]).reshape(-1, 1)
UREF_MAX = np.array([m * g / 2 + 1, m * g / 2 + 1]).reshape(-1, 1)


state_weights = np.array([1, 1, 1.0, 1.0, 1.0, 1.0])

STATE_MIN = np.concatenate((X_MIN.flatten(), X_MIN.flatten(), UREF_MIN.flatten()))
STATE_MAX = np.concatenate((X_MAX.flatten(), X_MAX.flatten(), UREF_MAX.flatten()))


class PvtolEnv(gym.Env):
    def __init__(self, sigma: float = 0.0):
        super(PvtolEnv, self).__init__()
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """
        self.num_dim_x = 6
        self.num_dim_control = 2
        self.pos_dimension = 2

        self.tracking_scaler = 1.0
        self.control_scaler = 0.0

        self.time_bound = 6.0
        self.dt = 0.03
        self.episode_len = int(self.time_bound / self.dt)
        self.t = np.arange(0, self.time_bound, self.dt)

        self.state_weights = state_weights
        self.sigma = sigma
        self.d_up = 3 * sigma

        self.Bbot_func = None
        self.effective_indices = np.arange(2, 4)

        self.use_learned_dynamics = False

        # initialize the ref trajectory
        self.x_0, _, _, _, self.xref, self.uref, self.episode_len = self.system_reset()
        self.init_tracking_error = np.linalg.norm(self.x_0 - self.xref[0], ord=2)

        self.disturbance_duration = 0.0
        self.current_disturbance = np.zeros(self.num_dim_x)

        self.observation_space = spaces.Box(
            low=STATE_MIN.flatten(), high=STATE_MAX.flatten(), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=UREF_MIN.flatten(), high=UREF_MAX.flatten(), dtype=np.float64
        )

    def replace_dynamics(self, dynamics_model: nn.Module):
        print("[INFO] The environment is now using learned dynamics for transition.")
        self.learned_dynamics_model = deepcopy(dynamics_model).cpu()
        self.learned_dynamics_model.device = torch.device("cpu")
        self.use_learned_dynamics = True

    def f_func(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            n = x.shape[0]

            p_x, p_z, phi, v_x, v_z, dot_phi = [x[:, i] for i in range(self.num_dim_x)]
            f = torch.zeros((n, self.num_dim_x))
            f[:, 0] = v_x * torch.cos(phi) - v_z * torch.sin(phi)
            f[:, 1] = v_x * torch.sin(phi) + v_z * torch.cos(phi)
            f[:, 2] = dot_phi
            f[:, 3] = v_z * dot_phi - g * torch.sin(phi)
            f[:, 4] = -v_x * dot_phi - g * torch.cos(phi)
            f[:, 5] = 0
        else:
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            n = x.shape[0]

            p_x, p_z, phi, v_x, v_z, dot_phi = [x[:, i] for i in range(self.num_dim_x)]
            f = np.zeros((n, self.num_dim_x))
            f[:, 0] = v_x * np.cos(phi) - v_z * np.sin(phi)
            f[:, 1] = v_x * np.sin(phi) + v_z * np.cos(phi)
            f[:, 2] = dot_phi
            f[:, 3] = v_z * dot_phi - g * np.sin(phi)
            f[:, 4] = -v_x * dot_phi - g * np.cos(phi)
            f[:, 5] = 0

        return f.squeeze()

    # def f_func(self, x):
    #     # x: bs x n x 1
    #     # f: bs x n x 1
    #     if len(x.shape) == 1:
    #         x = x.unsqueeze(0)
    #     n = x.shape[0]

    #     p_x, p_z, phi, v_x, v_z, dot_phi = [x[:, i] for i in range(self.num_dim_x)]
    #     f = torch.zeros((n, self.num_dim_x))
    #     f[:, 0] = v_x * torch.cos(phi) - v_z * torch.sin(phi)
    #     f[:, 1] = v_x * torch.sin(phi) + v_z * torch.cos(phi)
    #     f[:, 2] = dot_phi
    #     f[:, 3] = v_z * dot_phi - g * torch.sin(phi)
    #     f[:, 4] = -v_x * dot_phi - g * torch.cos(phi)
    #     f[:, 5] = 0
    #     return f

    # def B_func_np(self, x):
    #     if len(x.shape) == 1:
    #         x = x[np.newaxis, :]
    #     n = x.shape[0]

    #     B = np.zeros((n, self.num_dim_x, self.num_dim_control))

    #     B[:, 4, 0] = 1 / m
    #     B[:, 4, 1] = 1 / m
    #     B[:, 5, 0] = l / J
    #     B[:, 5, 1] = -l / J
    #     return B.squeeze()

    def B_func(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            n = x.shape[0]

            B = torch.zeros((n, self.num_dim_x, self.num_dim_control))

            B[:, 4, 0] = 1 / m
            B[:, 4, 1] = 1 / m
            B[:, 5, 0] = l / J
            B[:, 5, 1] = -l / J
        else:
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            n = x.shape[0]

            B = np.zeros((n, self.num_dim_x, self.num_dim_control))

            B[:, 4, 0] = 1 / m
            B[:, 4, 1] = 1 / m
            B[:, 5, 0] = l / J
            B[:, 5, 1] = -l / J
        return B.squeeze()

    def B_null(self, x: torch.Tensor | np.ndarray):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            n = x.shape[0]

            Bbot = torch.cat(
                (
                    torch.eye(
                        self.num_dim_x - self.num_dim_control,
                        self.num_dim_x - self.num_dim_control,
                    ),
                    torch.zeros(
                        (self.num_dim_control, self.num_dim_x - self.num_dim_control)
                    ),
                ),
                dim=0,
            )
            return Bbot.repeat(n, 1, 1).squeeze()
        else:
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            n = x.shape[0]

            Bbot = np.concatenate(
                (
                    np.eye(
                        self.num_dim_x - self.num_dim_control,
                        self.num_dim_x - self.num_dim_control,
                    ),
                    np.zeros(
                        (self.num_dim_control, self.num_dim_x - self.num_dim_control)
                    ),
                ),
                axis=0,
            )
            return np.repeat(Bbot[np.newaxis, :, :], n, axis=0).squeeze()

    def get_f_and_B(self, x: torch.Tensor | np.ndarray):
        if self.use_learned_dynamics:
            with torch.no_grad():
                f_x, B_x, Bbot_x = self.learned_dynamics_model(x)
            return (
                f_x.cpu().squeeze(0).numpy(),
                B_x.cpu().squeeze(0).numpy(),
                Bbot_x.cpu().squeeze(0).numpy(),
            )
        else:
            if self.Bbot_func is None:
                return self.f_func(x), self.B_func(x), self.B_null(x)
            else:
                return self.f_func(x), self.B_func(x), self.Bbot_func(x)

    def system_reset(self):
        # with temp_seed(int(seed)):
        xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (
            X_INIT_MAX - X_INIT_MIN
        )
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (
            XE_INIT_MAX - XE_INIT_MIN
        )
        x_0 = xref_0 + xe_0

        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (
            0.1 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))
        ).tolist()

        def sample_controls():
            uref = 0.5 * np.array([m * g, m * g])  # ref
            for freq, weight in zip(freqs, weights):
                uref += np.array(
                    [
                        weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                        weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    ]
                )
            uref = np.clip(uref, 0.75 * UREF_MIN.flatten(), 0.75 * UREF_MAX.flatten())

            u_mean = (UREF_MIN.flatten() + UREF_MAX.flatten()) / 2.0
            u_sigma = (UREF_MAX.flatten() - UREF_MIN.flatten()) / 18.0
            u = uref + np.random.normal(loc=u_mean, scale=u_sigma)
            u = np.clip(u, UREF_MIN.flatten(), UREF_MAX.flatten())

            return u, uref

        x_list, u_list, x_dot_list = [x_0], [], []
        xref_list, uref_list = [xref_0], []
        for i, _t in enumerate(self.t):
            u, uref = sample_controls()

            x_t = x_list[-1].copy()
            xref_t = xref_list[-1].copy()

            f_x, B_x = self.f_func(x_t), self.B_func(x_t)
            f_xref, B_xref = self.f_func(xref_t), self.B_func(xref_t)

            x_dot = f_x + np.matmul(B_x, u[:, np.newaxis]).squeeze()
            xref_dot = f_xref + np.matmul(B_xref, uref[:, np.newaxis]).squeeze()

            x_t = x_t + self.dt * x_dot
            xref_t = xref_t + self.dt * xref_dot

            termination1 = np.any(
                x_t[: self.pos_dimension] <= X_MIN.flatten()[: self.pos_dimension]
            ) or np.any(
                x_t[: self.pos_dimension] >= X_MAX.flatten()[: self.pos_dimension]
            )
            termination2 = np.any(
                xref_t[: self.pos_dimension] <= X_MIN.flatten()[: self.pos_dimension]
            ) or np.any(
                xref_t[: self.pos_dimension] >= X_MAX.flatten()[: self.pos_dimension]
            )
            termination = termination1 or termination2

            x_t = np.clip(x_t, X_MIN.flatten(), X_MAX.flatten())
            xref_t = np.clip(xref_t, X_MIN.flatten(), X_MAX.flatten())

            x_list.append(x_t)
            u_list.append(u)
            x_dot_list.append(x_dot)
            xref_list.append(xref_t)
            uref_list.append(uref)

            if termination:
                break

        return (
            x_0,
            np.array(x_list),
            np.array(u_list),
            np.array(x_dot_list),
            np.array(xref_list),
            np.array(uref_list),
            i,
        )

    def dynamic_fn(self, action):
        f_x, B_x, _ = self.get_f_and_B(self.x_t)

        self.x_t = self.x_t + self.dt * (
            f_x + np.matmul(B_x, action[:, np.newaxis]).squeeze()
        )

        if self.disturbance_duration <= 0.0:
            # duration in seconds
            self.disturbance_duration = np.random.uniform(0.0, 1.0)
            magnitude = np.random.uniform(0.0, self.sigma, size=self.num_dim_x)
            self.current_disturbance = (
                np.random.choice([-1.0, 1.0], size=self.num_dim_x) * magnitude
            )
            # self.current_disturbance[self.pos_dimension :] = (
            #     0.0  # position disturbance only
            # )

        if self.sigma > 0.0:
            self.x_t += self.current_disturbance
            self.disturbance_duration -= self.dt

        termination = np.any(
            self.x_t[: self.pos_dimension] <= X_MIN.flatten()[: self.pos_dimension]
        ) or np.any(
            self.x_t[: self.pos_dimension] >= X_MAX.flatten()[: self.pos_dimension]
        )
        self.x_t = np.clip(self.x_t, X_MIN.flatten(), X_MAX.flatten())

        # self.state = np.concatenate((self.x_t, self.xref[self.time_steps]))
        self.state = np.concatenate(
            (self.x_t, self.xref[self.time_steps], self.uref[self.time_steps])
        )
        self.time_steps += 1

        return termination

    def reward_fn(self, action):
        error = self.x_t - self.xref[self.time_steps]

        tracking_error = error.T @ error
        control_effort = np.linalg.norm(action, ord=2)

        rewards = (
            1 / (tracking_error + 1) + self.control_scaler / (control_effort + 1)
        ).squeeze(-1)

        return rewards, {
            "tracking_error": tracking_error,
            "control_effort": control_effort,
        }

    def reset(self, seed=None, options: dict | None = None):
        super().reset(seed=seed)
        self.time_steps = 0

        if options is None:
            self.x_0, _, _, _, self.xref, self.uref, self.episode_len = (
                self.system_reset()
            )
            self.init_tracking_error = np.linalg.norm(self.x_0 - self.xref[0], ord=2)
        else:
            if options.get("replace_x_0", True):
                xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (
                    XE_INIT_MAX - XE_INIT_MIN
                )
                x_0 = self.xref[0] + xe_0
                self.x_0 = x_0

                self.init_tracking_error = np.linalg.norm(
                    self.x_0 - self.xref[0], ord=2
                )

        self.x_t = self.x_0.copy()
        self.state = np.concatenate(
            (self.x_t, self.xref[self.time_steps], self.uref[self.time_steps])
        )
        # self.state = np.concatenate((self.x_t, self.xref[self.time_steps]))

        return self.state, {"x": self.x_t}

    def step(self, action):
        # policy output ranges [-1, 1]
        action = self.uref[self.time_steps] + action
        action = np.clip(action, UREF_MIN.flatten(), UREF_MAX.flatten())

        termination = self.dynamic_fn(action)
        reward, infos = self.reward_fn(action)

        truncation = self.time_steps == self.episode_len

        return (
            self.state,
            reward,
            termination,
            truncation,
            {
                "x": self.x_t,
                "tracking_error": infos["tracking_error"],
                "control_effort": infos["control_effort"],
                "relative_tracking_error": infos["tracking_error"]
                / self.init_tracking_error,
            },
        )

    def render(self, mode="human"):
        pass

    def get_rollout(self, buffer_size: int):
        """
        Mode: Specifies whether the rollout is for training or evaluation.
            - Offline: fully offline case where we use reference control to generate data.
        """
        data = dict(
            x=np.full(((buffer_size, self.num_dim_x)), np.nan, dtype=np.float32),
            u=np.full((buffer_size, self.num_dim_control), np.nan, dtype=np.float32),
            x_dot=np.full(((buffer_size, self.num_dim_x)), np.nan, dtype=np.float32),
            x_dot_true=np.full(
                ((buffer_size, self.num_dim_x)), np.nan, dtype=np.float32
            ),
            xref=np.full(((buffer_size, self.num_dim_x)), np.nan, dtype=np.float32),
            uref=np.full((buffer_size, self.num_dim_control), np.nan, dtype=np.float32),
        )

        num_samples = 0
        with tqdm(total=buffer_size, desc="Collecting samples", unit="samples") as pbar:
            while num_samples < buffer_size:
                _, x, u, x_dot_true, xref, uref, episode_len = self.system_reset()
                episode_len += 1

                biased_data = False
                if biased_data:
                    # Bias is 10% of the true value
                    biased_mean = 0.1 * x_dot + x_dot
                    # Variance is set so that 3σ + bias stays within ±20%
                    sigma = 0.1 * np.abs(x_dot) / 3.0

                    # Generate Gaussian noise with 10% bias
                    x_dot = np.random.normal(
                        loc=biased_mean, scale=sigma, size=x_dot_true.shape
                    )
                else:
                    x_dot = x_dot_true

                start_idx = num_samples
                end_idx = np.clip(start_idx + episode_len, 0, buffer_size)

                data["x"][start_idx:end_idx] = x[: end_idx - start_idx]
                data["u"][start_idx:end_idx] = u[: end_idx - start_idx]
                data["x_dot"][start_idx:end_idx] = x_dot[: end_idx - start_idx]
                data["x_dot_true"][start_idx:end_idx] = x_dot_true[
                    : end_idx - start_idx
                ]
                data["xref"][start_idx:end_idx] = xref[: end_idx - start_idx]
                data["uref"][start_idx:end_idx] = uref[: end_idx - start_idx]

                added_samples = end_idx - start_idx
                num_samples += added_samples
                pbar.update(added_samples)  # Update progress bar

        # Check for NaNs
        if np.any(np.isnan(data["x"])):
            print("NaN values found in x")

        return data
