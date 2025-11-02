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
XREF_INIT_MIN = np.array([0, 0, flapper_height, 0.0, 0.0, 0.0, 0.8, 0, 0, 0])
XREF_INIT_MAX = np.array([0, 0, flapper_height, 0.0, 0.0, 0.0, 0.8, 0, 0, 0])

# perturbation to the reference state
lim = 0.1
XE_INIT_MIN = np.array([-lim, -lim, 0, 0, 0, 0, 0, 0, 0, 0])  # .reshape(-1, 1)
XE_INIT_MAX = np.array([lim, lim, 0, 0, 0, 0, 0, 0, 0, np.pi / 2])  # .reshape(-1, 1)

# reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim]).reshape(
    -1, 1
)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim, lim, lim, lim, lim]).reshape(-1, 1)

# reference control bounds
UREF_MIN = np.array([-1.0, -50.0, -50.0, -50.0]).reshape(-1, 1)
UREF_MAX = np.array([1.0, 50.0, 50.0, 50.0]).reshape(-1, 1)


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
    "r": 0.1,  # control cost weight
}


class FlapperEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform"):
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "flapper"
        self.v = np.array(
            [
                0.4776,
                0.767,
                0.2251,
                -9.7609,
                -2.0317,
                -1.1271,
                0.0027,
                -0.0004,
                -0.0003,
                0.0002,
            ]
        )
        self.c = np.array(
            [
                0.002,
                0.0022,
                0.0216,
                0.1162,
                0.0021,
                10.2837,
                0.0012,
                0.0002,
                -0.0009,
                -0.0075,
            ]
        )

        # initialize the base environment
        env_config["sample_mode"] = sample_mode
        env_config["Bbot_func"] = None

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

        # application of grey-box model
        x_dot = self.v * (x_hat_dot) + self.c

        return x_dot

    def sample_reference_controls(self, freqs, weights, _t, add_noise=False):
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

    def system_reset(self):
        """Resets the system to an initial state and generates a reference trajectory."""
        xref_0, xe_0, x_0 = self.define_initial_state()

        # Generate reference trajectory
        freqs = [0.1 * i for i in range(1, 11)]  # flapper is vulnerable to high freq
        weights = np.random.randn(len(freqs), len(self.UREF_MIN))
        weights = (weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()

        xref_list, uref_list = [xref_0], []
        for i, _t in enumerate(self.t):
            uref_t = self.sample_reference_controls(freqs, weights, _t)
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
