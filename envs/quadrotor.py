import numpy as np
import torch

from envs.env_base import BaseEnv

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

# X bounds
X_MIN = np.array(
    [-30.0, -30.0, -30.0, -x4_lim, -x5_lim, -x6_lim, x7_low, -x8_lim, -x9_lim, -x10_lim]
).reshape(-1, 1)
X_MAX = np.array(
    [30.0, 30.0, 30.0, x4_lim, x5_lim, x6_lim, x7_high, x8_lim, x9_lim, x10_lim]
).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([-5, -5, -5, -1.0, -1.0, -1.0, g, 0, 0, 0])
XREF_INIT_MAX = np.array([5, 5, 5, 1.0, 1.0, 1.0, g, 0, 0, 0])

# Initial reference state perturbation bounds
XE_INIT_MIN = -0.5 * np.ones(10)
XE_INIT_MAX = 0.5 * np.ones(10)

# State perturbation bounds for c3m
lim = 1.0
XE_MIN = -lim * np.ones(10).reshape(-1, 1)
XE_MAX = lim * np.ones(10).reshape(-1, 1)

# reference control bounds
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
    "dt": 0.03,
    "time_bound": 6.0,
    "use_learned_dynamics": False,
    "q": 1.0,  # state cost weight
    "r": 0.1,  # control cost weight
}


class QuadRotorEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform"):
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "quadrotor"

        # initialize the base environment
        env_config["sample_mode"] = sample_mode

        super(QuadRotorEnv, self).__init__(env_config)

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
        f[:, 3] = -force * lib.sin(theta_y)
        f[:, 4] = force * lib.cos(theta_y) * lib.sin(theta_x)
        f[:, 5] = g - force * lib.cos(theta_y) * lib.cos(theta_x)
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

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        uref = np.array([0.0, 0.0, 0.0, 0.0])  # ref
        for freq, weight in zip(freqs, weights):
            uref += np.array(
                [
                    weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    weight[1] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    weight[2] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    weight[3] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                ]
            )
        if add_noise:
            # add gaussian noise
            uref += np.random.normal(0, np.abs(0.1 * uref), size=uref.shape)

        uref = np.clip(uref, UREF_MIN.flatten(), UREF_MAX.flatten())
        return uref

    def render(self, mode="human"):
        pass
