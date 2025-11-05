import numpy as np
import torch

from envs.env_base import BaseEnv

# CAR PARAMETERS
v_l = 1.0
v_h = 2.0

# X bounds
X_MIN = np.array([-15.0, -15.0, -np.pi, v_l]).reshape(-1, 1)
X_MAX = np.array([15.0, 15.0, np.pi, v_h]).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([-2.0, -2.0, -1.0, 1.5])
XREF_INIT_MAX = np.array([2.0, 2.0, 1.0, 1.5])

# Initial reference state perturbation bounds
XE_INIT_MIN = np.full((4,), -1.0)
XE_INIT_MAX = np.full((4,), 1.0)

# reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim]).reshape(-1, 1)

# reference control bounds
UREF_MIN = np.array([-3.0, -3.0]).reshape(-1, 1)
UREF_MAX = np.array([3.0, 3.0]).reshape(-1, 1)

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
    "num_dim_control": 2,
    "pos_dimension": 2,
    "dt": 0.03,
    "time_bound": 6.0,
    "use_learned_dynamics": False,
    "q": 1.0,  # state cost weight
    "r": 0.1,  # control cost weight
}


class CarEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform"):
        """
        State: tracking error between current and reference trajectory
        Reward: The 2-norm of tracking error
        """

        # env specific parameters
        self.task = "car"

        # initialize the base environment
        env_config["sample_mode"] = sample_mode

        super(CarEnv, self).__init__(env_config)

    def _f_logic(self, x, lib):
        """Calculates the f(x) vector using the provided library."""
        n = x.shape[0]
        f = lib.zeros((n, self.num_dim_x))

        f[:, 0] = x[:, 3] * lib.cos(x[:, 2])
        f[:, 1] = x[:, 3] * lib.sin(x[:, 2])
        return f

    def _B_logic(self, x, lib):
        """Calculates the B(x) matrix using the provided library."""
        n = x.shape[0]
        B = lib.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 2, 0] = 1
        B[:, 3, 1] = 1
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        uref = np.array([0.0, 0])
        for freq, weight in zip(freqs, weights):
            uref += np.array(
                [weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi), 0]
            )
        if add_noise:
            # add gaussian noise
            uref += np.random.normal(0, np.abs(0.1 * uref), size=uref.shape)

        uref = np.clip(uref, UREF_MIN.flatten(), UREF_MAX.flatten())
        return uref

    def render(self, mode="human"):
        pass
