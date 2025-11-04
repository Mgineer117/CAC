import gymnasium as gym
import numpy as np
import torch
from env_base import BaseEnv
from gymnasium import spaces

# X bounds
X_MIN = np.array([-5.0, -np.pi / 3, -1.0, -np.pi]).reshape(-1, 1)
X_MAX = np.array([5.0, np.pi / 3, 1.0, np.pi]).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([0.0, 0, 0.0, 0])
XREF_INIT_MAX = np.array([0.0, 0, 0.0, 0])

# Initial perturbation to the reference state
XE_INIT_MIN = np.array([-1.0, -np.pi / 3, -0.5, -np.pi])
XE_INIT_MAX = np.array([1.0, np.pi / 3, 0.5, np.pi])

# initial reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim]).reshape(-1, 1)

# reference control bounds
UREF_MIN = np.array(
    [
        -1.0,
    ]
).reshape(-1, 1)
UREF_MAX = np.array(
    [
        1.0,
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
    "time_bound": 6.0,
    "use_learned_dynamics": False,
    "q": 1.0,  # state cost weight
    "r": 0.1,  # control cost weight
}


class SegwayEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform"):
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "segway"

        # initialize the base environment
        env_config["sample_mode"] = sample_mode
        env_config["Bbot_func"] = self._B_null_logic

        super(SegwayEnv, self).__init__()

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

        # v_x_dot (state x[2])
        f[:, 2] = (
            lib.cos(theta) * (9.8 * lib.sin(theta) + 11.5 * v)
            + 68.4 * v
            - 1.2 * (omega**2) * lib.sin(theta)
        ) / (lib.cos(theta) - 24.7)

        # omega_dot (state x[3])
        f[:, 3] = (
            -58.8 * v * lib.cos(theta)
            - 243.5 * v
            - lib.sin(theta) * (208.3 + (omega**2) * lib.cos(theta))
        ) / (lib.cos(theta) ** 2 - 24.7)

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

        # v_x_dot term (state x[2])
        B[:, 2, 0] = (-1.8 * lib.cos(theta) - 10.9) / (lib.cos(theta) - 24.7)

        # omega_dot term (state x[3])
        B[:, 3, 0] = (9.3 * lib.cos(theta) + 38.6) / (lib.cos(theta) ** 2 - 24.7)

        return B

    def _B_null_logic(self, x, lib):
        """
        Calculates the orthogonal complement B_null(x) (or B_bot).
        This logic is taken from your 'Bbot_func'.
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(0) if lib == torch else x[np.newaxis, :]

        n = x.shape[0]
        p, theta, v, omega = [x[:, i] for i in range(self.num_dim_x)]

        # B_null has (num_dim_x - num_dim_control) = 3 columns
        B_null = lib.zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control))

        # Column 0 (corresponds to p_x)
        B_null[:, 0, 0] = 1.0

        # Column 1 (corresponds to theta)
        B_null[:, 1, 1] = 1.0

        # Column 2 (the complex one that makes B^T * B_null = 0)
        # B_null[:, 2, 2] = B4(x)
        B_null[:, 2, 2] = (9.3 * lib.cos(theta) + 38.6) / (lib.cos(theta) ** 2 - 24.7)

        # B_null[:, 3, 2] = -B3(x)
        B_null[:, 3, 2] = -(-1.8 * lib.cos(theta) - 10.9) / (lib.cos(theta) - 24.7)

        return B_null

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

        freqs = []
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (
            0.0 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))
        ).tolist()

        xref = [xref_0]
        uref = []
        for i, _t in enumerate(self.t):
            u = np.array([10.2 * xref_0[2] / 47.9])  # ref
            for freq, weight in zip(freqs, weights):
                u += np.array(
                    [
                        weight[0]
                        * (-1) ** (int(freq * _t / self.time_bound))
                        * np.sin(freq * _t / self.time_bound * 2 * np.pi),
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
            x_dot_true=np.full(
                ((buffer_size, self.num_dim_x)), np.nan, dtype=np.float32
            ),
            xref=np.full(((buffer_size, self.num_dim_x)), np.nan, dtype=np.float32),
            uref=np.full((buffer_size, self.num_dim_control), np.nan, dtype=np.float32),
        )

        for i in range(buffer_size):
            xref = np.random.uniform(X_MIN.flatten(), X_MAX.flatten())
            uref = np.random.uniform(UREF_MIN.flatten(), UREF_MAX.flatten())

            xe = np.random.uniform(XE_MIN.flatten(), XE_MAX.flatten())
            x = np.clip(xref + xe, X_MIN.flatten(), X_MAX.flatten())
            u = np.random.uniform(UREF_MIN.flatten(), UREF_MAX.flatten())

            x_dot_true = (
                self.f_func_np(x)
                + np.matmul(self.B_func_np(x), u[:, np.newaxis]).squeeze()
            )

            # Bias is 10% of the true value
            bias = 0.1 * x_dot_true

            # Variance is set so that 3σ + bias stays within ±20%
            sigma = 0.1 * np.abs(x_dot_true) / 3.0

            # Generate Gaussian noise with 10% bias and bounded 10% std dev
            noise = np.random.normal(loc=bias, scale=sigma, size=x_dot_true.shape)

            # Final noisy x_dot
            x_dot = x_dot_true + noise

            data["x"][i] = x
            data["u"][i] = u
            data["x_dot"][i] = x_dot
            data["x_dot_true"][i] = x_dot_true
            data["xref"][i] = xref
            data["uref"][i] = uref

        return data
