import numpy as np
import torch

from envs.env_base import BaseEnv

# TURTLEBOT PARAMETERS
k1, k2, k3 = 0.9061, 0.8831, 0.8548

# X bounds
X_MIN = np.array([-5.0, -2.0, 0]).reshape(-1, 1)
X_MAX = np.array([0.0, 2.0, 2 * np.pi]).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([-1.7, 0.75, np.pi])
XREF_INIT_MAX = np.array([-1.3, 1.15, (3 / 2) * np.pi])

# Initial reference state perturbation bounds
XE_INIT_MIN = np.array([-0.1, -0.1, -(1 / 4) * np.pi])
XE_INIT_MAX = np.array([0.1, 0.1, (1 / 4) * np.pi])

# reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim]).reshape(-1, 1)

# reference control bounds
UREF_MIN = np.array([0.0, -1.82]).reshape(-1, 1)
UREF_MAX = np.array([0.22, 1.82]).reshape(-1, 1)

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
    "num_dim_x": 3,
    "num_dim_control": 2,
    "pos_dimension": 2,
    "dt": 0.1,
    "time_bound": 20.0,
    "use_learned_dynamics": False,
    "q": 1.0,  # state cost weight
    "r": 0.1,  # control cost weight
}


class TurtlebotEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform"):
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "turtlebot"

        # initialize the base environment
        env_config["sample_mode"] = sample_mode
        super(TurtlebotEnv, self).__init__(env_config)

    def _f_logic(self, x, lib):
        """Calculates the f(x) vector using the provided library."""
        n = x.shape[0]
        p_x, p_z, theta = [x[:, i] for i in range(self.num_dim_x)]
        f = lib.zeros((n, self.num_dim_x))
        return f

    def _B_logic(self, x, lib):
        """Calculates the B(x) matrix using the provided library."""
        n = x.shape[0]
        p_x, p_y, theta = [x[:, i] for i in range(self.num_dim_x)]

        B = lib.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 0, 0] = k1 * lib.cos(theta)
        B[:, 1, 0] = k2 * lib.sin(theta)
        B[:, 2, 1] = k3
        return B

    def _B_null_logic(self, x, n, lib):
        """
        Calculates the orthogonal complement B_null(x) (or B_bot).
        This logic is taken from your 'Bbot_func'.
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(0) if lib == torch else x[np.newaxis, :]

        p_x, p_y, theta = [x[:, i] for i in range(self.num_dim_x)]

        Bbot = lib.zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control))

        Bbot[:, 0, 0] = k2 * lib.sin(theta) * k3
        Bbot[:, 1, 0] = -k1 * lib.cos(theta) * k3
        Bbot[:, 2, 0] = 0.0

        return Bbot

    # def theta_correction(self, x: np.ndarray):
    #     x[2] = np.mod(x[2], 2 * np.pi)
    #     return x

    # def system_reset(self):
    #     # with temp_seed(int(seed)):
    #     patterns = [
    #         "constant_linear_sin_angular",
    #         "changing_linear_sin_angular",
    #         "constant_linear_step_angular",
    #         "accelerating_spiral_out",
    #         "decelerating_spiral_in",
    #         "default",
    #     ]

    #     # Randomly choose a pattern
    #     selected_pattern = random.choice(patterns)

    #     xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (
    #         X_INIT_MAX - X_INIT_MIN
    #     )
    #     xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (
    #         XE_INIT_MAX - XE_INIT_MIN
    #     )
    #     x_0 = xref_0 + xe_0

    #     freqs = list(range(1, 11))
    #     weights = np.random.randn(len(freqs), len(UREF_MIN))
    #     weights = (weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()

    #     xref = [xref_0]
    #     uref = []
    #     for i, _t in enumerate(self.t):
    #         u = self.get_motion(_t, pattern=selected_pattern)
    #         # u = np.array([0.05, 0])  # ref
    #         # for freq, weight in zip(freqs, weights):
    #         #     u += np.array(
    #         #         [
    #         #             weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
    #         #             weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
    #         #         ]
    #         #     )
    #         u = np.clip(u, 0.75 * UREF_MIN.flatten(), 0.75 * UREF_MAX.flatten())

    #         x_t = xref[-1].copy()

    #         f_x = self.f_func_np(x_t)
    #         B_x = self.B_func_np(x_t)

    #         x_t = x_t + self.dt * (f_x + np.matmul(B_x, u[:, np.newaxis]).squeeze())

    #         termination = np.any(
    #             x_t[: self.pos_dimension] <= X_MIN.flatten()[: self.pos_dimension]
    #         ) or np.any(
    #             x_t[: self.pos_dimension] >= X_MAX.flatten()[: self.pos_dimension]
    #         )

    #         x_t = self.theta_correction(x_t)
    #         x_t = np.clip(x_t, X_MIN.flatten(), X_MAX.flatten())
    #         xref.append(x_t)
    #         uref.append(u)

    #         if termination:
    #             break

    #     return x_0, np.array(xref), np.array(uref), i

    # def get_motion(self, elapsed_time, pattern="constant_linear_sin_angular"):
    #     if pattern == "constant_linear_sin_angular":
    #         # 🚗 Constant speed, but turning left and right smoothly
    #         linear_velocity = 0.15
    #         angular_velocity = 0.4 * math.sin(0.5 * elapsed_time)

    #     elif pattern == "changing_linear_sin_angular":
    #         # 🚗 Speed up and slow down smoothly, while also turning
    #         linear_velocity = 0.1 + 0.05 * math.sin(
    #             0.3 * elapsed_time
    #         )  # speed oscillates between 0.05 and 0.15
    #         angular_velocity = 0.4 * math.sin(0.5 * elapsed_time)

    #     elif pattern == "constant_linear_step_angular":
    #         # 🚗 Constant speed, but angular velocity changes in a square wave (like zigzag)
    #         linear_velocity = 0.15
    #         angular_velocity = 0.4 * np.sign(math.sin(0.5 * elapsed_time))

    #     elif pattern == "accelerating_spiral_out":
    #         # 🚗 Linear velocity slowly increases, angular velocity decreases
    #         linear_velocity = 0.1 + 0.01 * elapsed_time  # slowly accelerating
    #         angular_velocity = 0.5 / (1 + elapsed_time)  # decreasing turn, spiral out

    #     elif pattern == "decelerating_spiral_in":
    #         # 🚗 Linear velocity slowly decreases, angular velocity increases
    #         linear_velocity = max(0.05, 0.15 - 0.01 * elapsed_time)
    #         angular_velocity = 0.1 + 0.05 * elapsed_time  # increasing turn, spiral in

    #     else:
    #         # Default fallback: straight line
    #         linear_velocity = 0.1
    #         angular_velocity = 0.0

    #     return np.array([linear_velocity, angular_velocity])

    def render(self, mode="human"):
        pass
