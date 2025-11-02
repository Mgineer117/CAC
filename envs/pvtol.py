import numpy as np
import torch

from envs.env_base import BaseEnv

# PVTOL PARAMETERS
m = 0.486
J = 0.00383
g = 9.81
l = 0.25

p_lim = np.pi / 3
pd_lim = np.pi / 3
vx_lim = 2.0
vz_lim = 1.0

# X bounds
X_MIN = np.array([-15.0, 0.0, -p_lim, -vx_lim, -vz_lim, -pd_lim]).reshape(-1, 1)
X_MAX = np.array([15.0, 10.0, p_lim, vx_lim, vz_lim, pd_lim]).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([-1, 5.0, -0.1, 0.5, 0.0, 0.0])
XREF_INIT_MAX = np.array([1, 6.0, 0.1, 1.0, 0.0, 0.0])

# Initial reference state perturbation bounds
XE_INIT_MIN = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
XE_INIT_MAX = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

# reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim]).reshape(-1, 1)

# reference control bounds
UREF_MIN = np.array([m * g / 2 - 1, m * g / 2 - 1]).reshape(-1, 1)
UREF_MAX = np.array([m * g / 2 + 1, m * g / 2 + 1]).reshape(-1, 1)

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
    "num_dim_x": 6,
    "num_dim_control": 2,
    "pos_dimension": 2,
    "dt": 0.03,
    "time_bound": 6.0,
    "use_learned_dynamics": False,
    "q": 1.0,  # state cost weight
    "r": 0.1,  # control cost weight
}


class PvtolEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform"):
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "pvtol"

        # initialize the base environment
        env_config["sample_mode"] = sample_mode
        env_config["Bbot_func"] = None

        super(PvtolEnv, self).__init__(env_config)

    def _f_logic(self, x, lib):
        """Calculates the f(x) vector using the provided library."""
        n = x.shape[0]
        p_x, p_z, phi, v_x, v_z, dot_phi = [x[:, i] for i in range(self.num_dim_x)]

        f = lib.zeros((n, self.num_dim_x))
        f[:, 0] = v_x * lib.cos(phi) - v_z * lib.sin(phi)
        f[:, 1] = v_x * lib.sin(phi) + v_z * lib.cos(phi)
        f[:, 2] = dot_phi
        f[:, 3] = v_z * dot_phi - g * lib.sin(phi)
        f[:, 4] = -v_x * dot_phi - g * lib.cos(phi)
        f[:, 5] = 0
        return f

    def _B_logic(self, x, lib):
        """Calculates the B(x) matrix using the provided library."""
        n = x.shape[0]
        B = lib.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 4, 0] = 1 / m
        B[:, 4, 1] = 1 / m
        B[:, 5, 0] = l / J
        B[:, 5, 1] = -l / J
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

    def sample_reference_controls(self, freqs, weights, _t, add_noise=False):
        uref = 0.5 * np.array([m * g, m * g])  # ref
        for freq, weight in zip(freqs, weights):
            uref += np.array(
                [
                    weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    weight[1] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                ]
            )
        if add_noise:
            # add gaussian noise
            uref += np.random.normal(0, np.abs(0.1 * uref), size=uref.shape)

        uref = np.clip(uref, UREF_MIN.flatten(), UREF_MAX.flatten())
        return uref

    def render(self, mode="human"):
        pass
