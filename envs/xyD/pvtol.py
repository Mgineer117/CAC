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
    "r": 0.01,  # control cost weight
}


class PvtolEnv(BaseEnv):
    def __init__(
        self,
        sample_mode: str = "uniform",
        reward_mode: str = "default",
        n_control_per_x: int = 1,
    ):
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "pvtol"

        # initialize the base environment
        env_config["sample_mode"] = sample_mode
        env_config["reward_mode"] = reward_mode
        env_config["n_control_per_x"] = n_control_per_x

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

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
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

    def system_reset(self):
        """Resets the system to an initial state and generates a reference trajectory."""
        xref_0, xe_0, x_0 = self.define_initial_state()

        # Generate reference trajectory
        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (
            0.1 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))
        ).tolist()

        xref_list, uref_list = [xref_0], []
        for i, _t in enumerate(self.t):
            uref_t = self.sample_reference_controls(
                freqs, weights, _t, {"xref_0": xref_0}
            )
            xref_t, term, trunc = self.get_transition(xref_list[-1].copy(), uref_t)

            xref_list.append(xref_t)
            uref_list.append(uref_t)

            if term or trunc:
                break

        return (
            x_0,
            np.array(xref_list),
            np.array(uref_list),
            i,
        )

    def render(self, mode="human"):
        pass
