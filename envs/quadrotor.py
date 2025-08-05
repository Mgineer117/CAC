import os
import urllib.request

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# QUADROTOR PARAMETERS
g = 9.81

x10_lim = np.pi / 3
x9_lim = np.pi / 3
x8_lim = np.pi / 3
x7_low = 0.5 * g
x7_high = 2 * g
x4_lim = 1.5
x5_lim = 1.5
x6_lim = 1.5

X_MIN = np.array(
    [-30.0, -30.0, -30.0, -x4_lim, -x5_lim, -x6_lim, x7_low, -x8_lim, -x9_lim, -x10_lim]
).reshape(-1, 1)
X_MAX = np.array(
    [30.0, 30.0, 30.0, x4_lim, x5_lim, x6_lim, x7_high, x8_lim, x9_lim, x10_lim]
).reshape(-1, 1)

# we noticed that the last item of u is dead and useless
UREF_MIN = np.array([-1.0, -1.0, -1.0, -1.0]).reshape(-1, 1)
UREF_MAX = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)

lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim]).reshape(
    -1, 1
)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim, lim, lim, lim, lim]).reshape(-1, 1)

# for sampling ref
X_INIT_MIN = np.array([-5, -5, -5, -1.0, -1.0, -1.0, g, 0, 0, 0])
X_INIT_MAX = np.array([5, 5, 5, 1.0, 1.0, 1.0, g, 0, 0, 0])

XE_INIT_MIN = np.array(
    [
        -0.5,
    ]
    * 10
)
XE_INIT_MAX = np.array(
    [
        0.5,
    ]
    * 10
)

# x, y, z, vx, vy, vz, force, theta_x, theta_y, theta_z
# state_weights = np.array([1, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# state_weights = np.array([1, 1, 1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1])
# state_weights = np.array([1, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
# state_weights = np.array([1, 1, 1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
state_weights = np.array([1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

STATE_MIN = np.concatenate((X_MIN.flatten(), X_MIN.flatten(), UREF_MIN.flatten()))
STATE_MAX = np.concatenate((X_MAX.flatten(), X_MAX.flatten(), UREF_MAX.flatten()))


class QuadRotorEnv(gym.Env):
    def __init__(self, sigma: float = 0.0):
        super(QuadRotorEnv, self).__init__()
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """
        self.num_dim_x = 10
        self.num_dim_control = 4
        self.pos_dimension = 3

        self.tracking_scaler = 1.0
        self.control_scaler = 0.0

        self.time_bound = 6.0
        self.dt = 0.03
        self.episode_len = int(self.time_bound / self.dt)
        self.t = np.arange(0, self.time_bound, self.dt)

        self.state_weights = state_weights
        self.sigma = sigma
        self.d_up = 3 * sigma

        self.effective_indices = np.arange(3, 9)
        self.Bbot_func = None

        # initialize the ref trajectory
        self.x_0, self.xref, self.uref, self.episode_len = self.system_reset()
        self.init_tracking_error = np.linalg.norm(self.x_0 - self.xref[0], ord=2)

        self.disturbance_duration = 0.0
        self.current_disturbance = np.zeros(self.num_dim_x)

        self.observation_space = spaces.Box(
            low=STATE_MIN.flatten(), high=STATE_MAX.flatten(), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=UREF_MIN.flatten(), high=UREF_MAX.flatten(), dtype=np.float64
        )

    def f_func_np(self, x):
        # x: bs x n x 1
        # f: bs x n x 1
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        n = x.shape[0]

        x, y, z, vx, vy, vz, force, theta_x, theta_y, theta_z = [
            x[:, i] for i in range(self.num_dim_x)
        ]
        f = np.zeros((n, self.num_dim_x))
        f[:, 0] = vx
        f[:, 1] = vy
        f[:, 2] = vz
        f[:, 3] = -force * np.sin(theta_y)
        f[:, 4] = force * np.cos(theta_y) * np.sin(theta_x)
        f[:, 5] = g - force * np.cos(theta_y) * np.cos(theta_x)
        f[:, 6] = 0
        f[:, 7] = 0
        f[:, 8] = 0
        f[:, 9] = 0

        return f.squeeze()

    def f_func(self, x):
        # x: bs x n x 1
        # f: bs x n x 1
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        n = x.shape[0]

        x, y, z, vx, vy, vz, force, theta_x, theta_y, theta_z = [
            x[:, i] for i in range(self.num_dim_x)
        ]
        f = torch.zeros((n, self.num_dim_x))
        f[:, 0] = vx
        f[:, 1] = vy
        f[:, 2] = vz
        f[:, 3] = -force * torch.sin(theta_y)
        f[:, 4] = force * torch.cos(theta_y) * torch.sin(theta_x)
        f[:, 5] = g - force * torch.cos(theta_y) * torch.cos(theta_x)
        f[:, 6] = 0
        f[:, 7] = 0
        f[:, 8] = 0
        f[:, 9] = 0

        return f  # .squeeze()

    def B_func_np(self, x):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        n = x.shape[0]

        B = np.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 6, 0] = 1
        B[:, 7, 1] = 1
        B[:, 8, 2] = 1
        B[:, 9, 3] = 1
        return B.squeeze()

    def B_func(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        n = x.shape[0]

        B = torch.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 6, 0] = 1
        B[:, 7, 1] = 1
        B[:, 8, 2] = 1
        B[:, 9, 3] = 1
        return B  # .squeeze()

    def B_null(self, x: torch.Tensor):
        n = x.shape[0]
        Bbot = torch.cat(
            (
                torch.eye(
                    self.num_dim_x - self.num_dim_control,
                    self.num_dim_x - self.num_dim_control,
                ),
                torch.zeros(
                    self.num_dim_control, self.num_dim_x - self.num_dim_control
                ),
            ),
            dim=0,
        )
        return Bbot.repeat(n, 1, 1)

    def get_f_and_B(self, x: torch.Tensor):
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
            2.0 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))
        ).tolist()

        xref = [xref_0]
        uref = []
        for i, _t in enumerate(self.t):
            u = np.array([0.0, 0.0, 0.0, 0.0])  # ref
            for freq, weight in zip(freqs, weights):
                u += np.array(
                    [
                        weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                        weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                        weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                        weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    ]
                )
            u = np.clip(u, 0.75 * UREF_MIN.flatten(), 0.75 * UREF_MAX.flatten())

            x_t = xref[-1].copy()

            f_x = self.f_func_np(x_t)
            B_x = self.B_func_np(x_t)

            x_t = x_t + self.dt * (f_x + np.matmul(B_x, u[:, np.newaxis]).squeeze())

            termination = np.any(
                x_t[: self.pos_dimension] <= X_MIN.flatten()[: self.pos_dimension]
            ) or np.any(
                x_t[: self.pos_dimension] >= X_MAX.flatten()[: self.pos_dimension]
            )

            x_t = np.clip(x_t, X_MIN.flatten(), X_MAX.flatten())
            xref.append(x_t)
            uref.append(u)

            if termination:
                break

        return x_0, np.array(xref), np.array(uref), i

    def dynamic_fn(self, action):
        self.time_steps += 1

        f_x = self.f_func_np(self.x_t)
        B_x = self.B_func_np(self.x_t)

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
            self.current_disturbance[self.pos_dimension :] = (
                0.0  # position disturbance only
            )

        if self.sigma > 0.0:
            self.x_t += self.current_disturbance
            self.disturbance_duration -= self.dt

        termination = np.any(
            self.x_t[: self.pos_dimension] <= X_MIN.flatten()[: self.pos_dimension]
        ) or np.any(
            self.x_t[: self.pos_dimension] >= X_MAX.flatten()[: self.pos_dimension]
        )
        self.x_t = np.clip(self.x_t, X_MIN.flatten(), X_MAX.flatten())
        self.state = np.concatenate(
            (self.x_t, self.xref[self.time_steps], self.uref[self.time_steps])
        )

        return termination

    def reward_fn(self, action):
        error = self.x_t - self.xref[self.time_steps]

        tracking_error = np.linalg.norm(
            self.state_weights * error,
            ord=2,
        )
        control_effort = np.linalg.norm(action, ord=2)

        reward = self.tracking_scaler / (tracking_error + 1) + self.control_scaler / (
            control_effort + 1
        )

        # reward = (self.time_steps / self.episode_len) * reward

        return reward, {
            "tracking_error": tracking_error,
            "control_effort": control_effort,
        }

    def reset(self, seed=None, options: dict | None = None):
        super().reset(seed=seed)
        self.time_steps = 0

        if options is None:
            self.x_0, self.xref, self.uref, self.episode_len = self.system_reset()
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
        data = dict(
            x=np.full(((buffer_size, self.num_dim_x)), np.nan, dtype=np.float32),
            u=np.full((buffer_size, self.num_dim_control), np.nan, dtype=np.float32),
            x_dot=np.full(((buffer_size, self.num_dim_x)), np.nan, dtype=np.float32),
            xref=np.full(((buffer_size, self.num_dim_x)), np.nan, dtype=np.float32),
            uref=np.full((buffer_size, self.num_dim_control), np.nan, dtype=np.float32),
        )

        for i in range(buffer_size):
            xref = np.random.uniform(X_MIN.flatten(), X_MAX.flatten())
            uref = np.random.uniform(UREF_MIN.flatten(), UREF_MAX.flatten())

            xe = np.random.uniform(XE_MIN.flatten(), XE_MAX.flatten())
            x = np.clip(xref + xe, X_MIN.flatten(), X_MAX.flatten())
            u = np.random.uniform(UREF_MIN.flatten(), UREF_MAX.flatten())

            x_dot = (
                self.f_func_np(x)
                + np.matmul(self.B_func_np(x), u[:, np.newaxis]).squeeze()
            )

            data["x"][i] = x
            data["u"][i] = u
            data["x_dot"][i] = x_dot
            data["xref"][i] = xref
            data["uref"][i] = uref

        return data
