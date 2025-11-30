import numpy as np
import torch

from envs.env_base import BaseEnv

# CARTPOLE PARAMETERS
mc = 1.0
mp = 1.0
g = 9.81
l = 1.0

# X bounds
X_MIN = np.array([-5.0, -np.pi / 3, -1.0, -1]).reshape(-1, 1)
X_MAX = np.array([5.0, np.pi / 3, 1.0, 1]).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([0.0, 0, 0.0, 0])
XREF_INIT_MAX = np.array([0.0, 0, 0.0, 0])

# Initial perturbation to the reference state
lim = 0.3
XE_INIT_MIN = np.array([-lim, -lim, -lim, -lim])
XE_INIT_MAX = np.array([lim, lim, lim, lim])

# initial reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim]).reshape(-1, 1)

# reference control bounds
UREF_MIN = np.array(
    [
        -3.0,
    ]
).reshape(-1, 1)
UREF_MAX = np.array(
    [
        3.0,
    ]
).reshape(-1, 1)

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
    "num_dim_x": 4,
    "num_dim_control": 1,
    "pos_dimension": 1,
    "dt": 0.03,
    "time_bound": 10.0,
    "use_learned_dynamics": False,
    "q": 1.0,  # state cost weight
    "r": 0.1,  # control cost weight
}


class CartPoleEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform", reward_mode: str = "default"):
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "cartpole"

        # initialize the base environment
        env_config["sample_mode"] = sample_mode
        env_config["reward_mode"] = reward_mode

        super(CartPoleEnv, self).__init__(env_config)

    def _f_logic(self, x, lib):
        """
        Calculates the drift dynamics f(x).
        This logic is taken from your 'f_func_np'.
        """
        # Ensure x is 2D (batch_size, num_dim_x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0) if lib == torch else x[np.newaxis, :]

        n = x.shape[0]
        p, theta, v, omega = [x[:, i] for i in range(self.num_dim_x)]

        f = lib.zeros((n, self.num_dim_x))
        f[:, 0] = v
        f[:, 1] = omega
        f[:, 2] = (
            mp
            * lib.sin(theta)
            * (l * (omega**2) - g * lib.cos(theta))
            / (mc + mp * (lib.sin(theta) ** 2))
        )
        f[:, 3] = (
            (
                mp * l * (omega**2) * lib.cos(theta) * lib.sin(theta)
                - (mc + mp) * g * lib.sin(theta)
            )
            / l
            / (mc + mp * (lib.sin(theta) ** 2))
        )

        return f

    def _B_logic(self, x, lib):
        """
        Calculates the control matrix B(x).
        This logic is taken from your 'B_func'.
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(0) if lib == torch else x[np.newaxis, :]

        n = x.shape[0]
        p, theta, v, omega = [x[:, i] for i in range(self.num_dim_x)]

        B = lib.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 2, 0] = 1 / (mc + mp * (lib.sin(theta) ** 2))
        B[:, 3, 0] = lib.cos(theta) / l / (mc + mp * (lib.sin(theta) ** 2))

        return B

    def _B_null_logic(self, x, n, lib):
        """
        Calculates the orthogonal complement B_null(x) (or B_bot).
        This logic is taken from your 'Bbot_func'.
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(0) if lib == torch else x[np.newaxis, :]

        p, theta, v, omega = [x[:, i] for i in range(self.num_dim_x)]

        # B_null has (num_dim_x - num_dim_control) = 3 columns
        B_null = lib.zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control))

        B_null[:, 0, 0] = 1
        B_null[:, 1, 1] = 1
        B_null[:, 2, 2] = lib.cos(theta) / l / (mc + mp * (lib.sin(theta) ** 2))
        B_null[:, 3, 2] = -1 / (mc + mp * (lib.sin(theta) ** 2))

        return B_null

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        xref_0 = infos["xref_0"]
        uref = np.array([10.2 * xref_0[2] / 47.9])  # ref
        for freq, weight in zip(freqs, weights):
            uref += np.array(
                [
                    weight[0]
                    * (-1) ** (int(freq * _t / self.time_bound))
                    * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                ]
            )
        if add_noise:
            # add gaussian noise
            uref += np.random.normal(0, np.abs(0.1 * uref), size=uref.shape)
        uref = np.clip(uref, UREF_MIN.flatten(), UREF_MAX.flatten())
        return uref
