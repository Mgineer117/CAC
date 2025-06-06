import math
import os
import random
import urllib.request

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# Truetlebot PARAMETERS
X_MIN = np.array([-5.0, -2.0, 0]).reshape(-1, 1)
X_MAX = np.array([0.0, 2.0, 2 * np.pi]).reshape(-1, 1)

k1, k2, k3 = 0.9061, 0.8831, 0.8548

lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim]).reshape(-1, 1)

# for sampling ref
X_INIT_MIN = np.array([-1.7, 0.75, np.pi])
X_INIT_MAX = np.array([-1.3, 1.15, (3 / 2) * np.pi])

XE_INIT_MIN = np.array([-0.1, -0.1, -(1 / 4) * np.pi])
XE_INIT_MAX = np.array([0.1, 0.1, (1 / 4) * np.pi])

UREF_MIN = np.array([0.0, -1.82]).reshape(-1, 1)
UREF_MAX = np.array([0.22, 1.82]).reshape(-1, 1)

state_weights = np.array([1, 1, 1])

STATE_MIN = np.concatenate((X_MIN.flatten(), X_MIN.flatten(), UREF_MIN.flatten()))
STATE_MAX = np.concatenate((X_MAX.flatten(), X_MAX.flatten(), UREF_MAX.flatten()))


class TurtlebotEnv(gym.Env):
    def __init__(self, sigma: float = 0.0):
        super(TurtlebotEnv, self).__init__()
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """
        self.num_dim_x = 3
        self.num_dim_control = 2
        self.pos_dimension = 2

        self.tracking_scaler = 1.0
        self.control_scaler = 0.0

        self.time_bound = 20.0
        self.dt = 0.1
        self.episode_len = int(self.time_bound / self.dt)
        self.t = np.arange(0, self.time_bound, self.dt)

        self.state_weights = state_weights
        self.sigma = sigma
        self.d_up = 3 * sigma

        self.effective_indices = np.arange(0, 3)

        self.observation_space = spaces.Box(
            low=STATE_MIN.flatten(), high=STATE_MAX.flatten(), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=UREF_MIN.flatten(), high=UREF_MAX.flatten(), dtype=np.float64
        )

    def Bbot_func(self, x: torch.Tensor):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        n = x.shape[0]
        theta = x[:, 2]  # assuming theta is at index 2

        Bbot = torch.zeros(
            n, self.num_dim_x, self.num_dim_x - self.num_dim_control
        ).type_as(x)

        Bbot[:, 0, 0] = k2 * torch.sin(theta) * k3
        Bbot[:, 1, 0] = -k1 * torch.cos(theta) * k3
        Bbot[:, 2, 0] = 0.0

        return Bbot

    def f_func_np(self, x):
        # x: bs x n x 1
        # f: bs x n x 1
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        n = x.shape[0]

        p_x, p_z, theta = [x[:, i] for i in range(self.num_dim_x)]
        f = np.zeros((n, self.num_dim_x))
        return f.squeeze()

    def f_func(self, x):
        # x: bs x n x 1
        # f: bs x n x 1
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        n = x.shape[0]

        p_x, p_z, theta = [x[:, i] for i in range(self.num_dim_x)]
        f = torch.zeros((n, self.num_dim_x))
        return f

    def B_func_np(self, x):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        n = x.shape[0]

        p_x, p_y, theta = [x[:, i] for i in range(self.num_dim_x)]

        B = np.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 0, 0] = k1 * np.cos(theta)
        B[:, 1, 0] = k2 * np.sin(theta)
        B[:, 2, 1] = k3

        return B.squeeze()

    def B_func(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        n = x.shape[0]

        p_x, p_z, theta = [x[:, i] for i in range(self.num_dim_x)]

        B = torch.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 0, 0] = k1 * torch.cos(theta)
        B[:, 1, 0] = k2 * torch.sin(theta)
        B[:, 2, 1] = k3
        return B

    def theta_correction(self, x: np.ndarray):
        x[2] = np.mod(x[2], 2 * np.pi)
        return x

    def system_reset(self):
        # with temp_seed(int(seed)):
        patterns = [
            "constant_linear_sin_angular",
            "changing_linear_sin_angular",
            "constant_linear_step_angular",
            "accelerating_spiral_out",
            "decelerating_spiral_in",
            "default",
        ]

        # Randomly choose a pattern
        selected_pattern = random.choice(patterns)

        xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (
            X_INIT_MAX - X_INIT_MIN
        )
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (
            XE_INIT_MAX - XE_INIT_MIN
        )
        x_0 = xref_0 + xe_0

        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()

        xref = [xref_0]
        uref = []
        for i, _t in enumerate(self.t):
            u = self.get_motion(_t, pattern=selected_pattern)
            # u = np.array([0.05, 0])  # ref
            # for freq, weight in zip(freqs, weights):
            #     u += np.array(
            #         [
            #             weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
            #             weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
            #         ]
            #     )
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

            x_t = self.theta_correction(x_t)
            x_t = np.clip(x_t, X_MIN.flatten(), X_MAX.flatten())
            xref.append(x_t)
            uref.append(u)

            if termination:
                break

        return x_0, np.array(xref), np.array(uref), i

    def get_motion(self, elapsed_time, pattern="constant_linear_sin_angular"):
        if pattern == "constant_linear_sin_angular":
            # 🚗 Constant speed, but turning left and right smoothly
            linear_velocity = 0.15
            angular_velocity = 0.4 * math.sin(0.5 * elapsed_time)

        elif pattern == "changing_linear_sin_angular":
            # 🚗 Speed up and slow down smoothly, while also turning
            linear_velocity = 0.1 + 0.05 * math.sin(
                0.3 * elapsed_time
            )  # speed oscillates between 0.05 and 0.15
            angular_velocity = 0.4 * math.sin(0.5 * elapsed_time)

        elif pattern == "constant_linear_step_angular":
            # 🚗 Constant speed, but angular velocity changes in a square wave (like zigzag)
            linear_velocity = 0.15
            angular_velocity = 0.4 * np.sign(math.sin(0.5 * elapsed_time))

        elif pattern == "accelerating_spiral_out":
            # 🚗 Linear velocity slowly increases, angular velocity decreases
            linear_velocity = 0.1 + 0.01 * elapsed_time  # slowly accelerating
            angular_velocity = 0.5 / (1 + elapsed_time)  # decreasing turn, spiral out

        elif pattern == "decelerating_spiral_in":
            # 🚗 Linear velocity slowly decreases, angular velocity increases
            linear_velocity = max(0.05, 0.15 - 0.01 * elapsed_time)
            angular_velocity = 0.1 + 0.05 * elapsed_time  # increasing turn, spiral in

        else:
            # Default fallback: straight line
            linear_velocity = 0.1
            angular_velocity = 0.0

        return np.array([linear_velocity, angular_velocity])

    def dynamic_fn(self, action):
        self.time_steps += 1

        f_x = self.f_func_np(self.x_t)
        B_x = self.B_func_np(self.x_t)

        self.x_t = self.x_t + self.dt * (
            f_x + np.matmul(B_x, action[:, np.newaxis]).squeeze()
        )

        noise = np.random.normal(loc=0.0, scale=self.sigma, size=self.num_dim_x)
        noise[self.pos_dimension :] = 0.0
        noise = np.clip(noise, -self.d_up, self.d_up)

        self.x_t += noise
        self.x_t = self.theta_correction(self.x_t)

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
