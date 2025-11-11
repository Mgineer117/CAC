import json
import os
import random
from math import ceil

import numpy as np
import torch

from envs.env_base import BaseEnv

# FLAPPER PARAMETERS
g = 9.81
flapper_height = 0.26  # height of the active marker

x4_lim = 1.0
x5_lim = 1.0
x6_lim = 1.0
x7_low = 0.0  # 0.5 * g
x7_high = 1.0  # 2 * g
x8_lim = np.pi / 3
x9_lim = np.pi / 3
x10_lim = np.pi / 3

# X bounds
X_MIN = np.array(
    [-2.0, -2.0, 0.0, -x4_lim, -x5_lim, -x6_lim, x7_low, -x8_lim, -x9_lim, -x10_lim]
).reshape(-1, 1)
X_MAX = np.array(
    [2.0, 2.0, 2.0, x4_lim, x5_lim, x6_lim, x7_high, x8_lim, x9_lim, x10_lim]
).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([0, 0, flapper_height, 0.0, 0.0, 0.0, 0.6, 0, 0, 0])
XREF_INIT_MAX = np.array([0, 0, flapper_height, 0.0, 0.0, 0.0, 0.6, 0, 0, 0])

# perturbation to the reference state
lim = 0.15
XE_INIT_MIN = np.array(
    [-lim, -lim, -0.05, -0.1, -0.1, -0.1, -0.05, 0, 0, 0]
)  # .reshape(-1, 1)
XE_INIT_MAX = np.array(
    [lim, lim, 0.05, 0.1, 0.1, 0.1, 0.05, 0.0, 0, np.pi / 2]
)  # .reshape(-1, 1)

# reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim]).reshape(
    -1, 1
)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim, lim, lim, lim, lim]).reshape(-1, 1)

# reference control bounds
# 2500 thrusts / 0.05s and 1.5 degs / 0.05 s
UREF_MIN = np.array([-1.0, -1.0, -1.0, -1.0]).reshape(-1, 1)
UREF_MAX = np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1)


env_config = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    "xref_init_min": XREF_INIT_MIN,
    "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN,
    "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN,
    "xe_max": XE_MAX,
    "uref_min": UREF_MIN,
    "uref_max": UREF_MAX,
    "num_dim_x": 10,
    "num_dim_control": 4,
    "pos_dimension": 3,
    "dt": 0.05,
    "time_bound": 30.0,
    "use_learned_dynamics": False,
    "q": 1.0,  # state cost weight
    "r": 0.3,  # control cost weight
}


class FlapperEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform", eval_env: bool = False):
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "flapper"
        self.eval_env = eval_env
        self.v = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.6567,
                1.6321,
                6.5546,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )
        self.c = np.array(
            [
                0.0,
                0.0,
                0.0,
                -0.0336,
                0.0101,
                -3.1939,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        self.std = np.array(
            [0.0, 0.0, 0.00, 0.5347, 0.3999, 0.2258, 0.0, 0.0, 0.0, 0.0]
        )

        # initialize the base environment
        env_config["Bbot_func"] = None
        env_config["sample_mode"] = sample_mode

        super(FlapperEnv, self).__init__(env_config)

    def _f_logic(self, x, lib):
        """Calculates the f(x) vector using the provided library."""
        n = x.shape[0]
        x, y, z, vx, vy, vz, force, theta_x, theta_y, theta_z = [
            x[:, i] for i in range(self.num_dim_x)
        ]
        f = lib.zeros((n, self.num_dim_x))
        f[:, 0] = vx
        f[:, 1] = vy
        f[:, 2] = vz
        # vx_dot (state 3) depends on ROLL (theta_x) and PITCH (theta_y)
        f[:, 3] = force * lib.cos(theta_y) * lib.sin(theta_x)

        # vy_dot (state 4) depends on PITCH (theta_y)
        f[:, 4] = -force * lib.sin(theta_y)

        # vz_dot (state 5) depends on ROLL (theta_x) and PITCH (theta_y)
        # This is the unitless regressor, (NO 'g'!)
        f[:, 5] = force * lib.cos(theta_y) * lib.cos(theta_x)
        # f[:, 5] = g - force * lib.cos(theta_y) * lib.cos(theta_x)
        f[:, 6] = 0
        f[:, 7] = 0
        f[:, 8] = 0
        f[:, 9] = 0

        return f

    def _B_logic(self, x, lib):
        """Calculates the B(x) matrix using the provided library."""
        n = x.shape[0]
        B = lib.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 6, 0] = 1
        B[:, 7, 1] = 1
        B[:, 8, 2] = 1
        B[:, 9, 3] = 1
        return B

    def _B_null_logic(self, n, lib):
        """Builds the B_null matrix batch using the provided library."""

        # Calculate the dimensions for the component matrices
        eye_dims = self.num_dim_x - self.num_dim_control
        zero_dims = (self.num_dim_control, eye_dims)

        if lib == torch:
            # 1. Create the base 2D matrix
            Bbot = torch.cat(
                (torch.eye(eye_dims), torch.zeros(zero_dims)),
                dim=0,
            )
            # 2. Repeat it 'n' times to create a 3D batch
            return Bbot.repeat(n, 1, 1)
        else:  # lib == np
            # 1. Create the base 2D matrix
            Bbot = np.concatenate(
                (np.eye(eye_dims), np.zeros(zero_dims)),
                axis=0,
            )
            # 2. Repeat it 'n' times to create a 3D batch
            #    (np.newaxis adds the first dimension for repeating)
            return np.repeat(Bbot[np.newaxis, :, :], n, axis=0)

    def get_dynamics(self, x: np.ndarray, u: np.ndarray):
        """Compute the dynamics x_dot given current state x and action u."""
        f_x, B_x, _ = self.get_f_and_B(x)
        x_hat_dot = f_x + np.matmul(B_x, u[:, np.newaxis]).squeeze()

        # application of grey-box model (deterministic part)
        x_dot_deterministic = self.v * (x_hat_dot) + self.c

        # add process noise
        # np.random.randn() creates samples from a standard normal distribution (mean 0, std 1)
        # We scale these samples by our learned std (self.std)
        process_noise = np.random.randn(*self.std.shape) * self.std

        # The final stochastic dynamics
        x_dot = x_dot_deterministic + process_noise

        return x_dot

    def system_reset(self):
        """Resets the system to an initial state and generates a reference trajectory."""
        # call json files in data/train/ using os
        if self.eval_env:
            json_file = "system_identifications/data/mdp_data/test_data.json"
        else:
            json_file = "system_identifications/data/mdp_data/train_data.json"

        # read json
        with open(json_file, "r") as file:
            data = json.load(file)

        num_traj = len(data["states"])

        # randomly pick one between 0 and num_traj -1
        traj_index = random.randint(0, num_traj - 1)

        states = data["states"][traj_index]
        actions = data["actions"][traj_index]

        _, xe_0, _ = self.define_initial_state()
        x_0 = np.array(states[0]) + xe_0

        return x_0, np.array(states), np.array(actions), self.episode_len

    def render(self, mode="human"):
        pass

    def get_rollout(self, buffer_size: int, mode: str):
        """
        Mode: Specifies whether the rollout is for training or evaluation.
            - Offline: fully offline case where we use reference control to generate data.
        """
        if mode == "c3m":
            c3m_data = dict(
                x=np.full((buffer_size, self.num_dim_x), np.nan, dtype=np.float32),
                xref=np.full((buffer_size, self.num_dim_x), np.nan, dtype=np.float32),
                uref=np.full(
                    (buffer_size, self.num_dim_control), np.nan, dtype=np.float32
                ),
            )

            # Sample all references at once
            xref = (self.X_MAX - self.X_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_x
            ) + self.X_MIN.flatten()
            uref = (self.UREF_MAX - self.UREF_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_control
            ) + self.UREF_MIN.flatten()
            xe = (self.XE_MAX - self.XE_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_x
            ) + self.XE_MIN.flatten()

            # Compose states
            x = xe + xref
            x = np.clip(x, self.X_MIN.flatten(), self.X_MAX.flatten())

            # Store
            c3m_data["x"] = x.astype(np.float32)
            c3m_data["xref"] = xref.astype(np.float32)
            c3m_data["uref"] = uref.astype(np.float32)

            # Check for NaNs
            if np.any(np.isnan(c3m_data["x"])):
                print("NaN values found in x")

            return c3m_data

        else:
            print(
                "\n[INFO] Flapper Env only supports trajectory as data collection method."
            )
            # call json files in data/train/ using os
            json_file = "system_identifications/data/mdp_data/train_data.json"

            # read json
            with open(json_file, "r") as file:
                data = json.load(file)

            # concat all trajectories
            x = np.concatenate(data["states"], axis=0)
            u = np.concatenate(data["actions"], axis=0)
            x_dot = np.concatenate(data["dynamics"], axis=0)

        return dict(x=x, u=u, x_dot=x_dot)
