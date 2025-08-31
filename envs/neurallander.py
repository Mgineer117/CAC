from copy import deepcopy
from math import ceil

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from tqdm import tqdm

# NEURAL-LANDER PARAMETERS
rho = 1.225
drone_height = 0.09
g = 9.81
mass = 1.47

X_MIN = np.array([-10.0, -10.0, 0.0, -1.0, -1.0, -1.0]).reshape(-1, 1)
X_MAX = np.array([10.0, 10.0, 3.0, 1.0, 1.0, 1.0]).reshape(-1, 1)

lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim]).reshape(-1, 1)

UREF_MIN = np.array([-1.0, -1.0, -3.0]).reshape(-1, 1)
UREF_MAX = np.array([1.0, 1.0, 9.0]).reshape(-1, 1)

# for sampling ref
X_INIT_MIN = np.array([-3.0, -3.0, 0.5, 1.0, 0.0, 0.0])
X_INIT_MAX = np.array([3.0, 3.0, 1.0, 1.0, 0.0, 0.0])

XE_INIT_MIN = np.array([-1, -1, -0.4, -1.0, -1.0, 0.0])
XE_INIT_MAX = np.array([1, 1.0, 1.0, 1.0, 1.0, 0.0])


state_weights = np.array([1, 1, 1, 1.0, 1.0, 1.0])

STATE_MIN = np.concatenate((X_MIN.flatten(), X_MIN.flatten(), UREF_MIN.flatten(), [0]))
STATE_MAX = np.concatenate((X_MAX.flatten(), X_MAX.flatten(), UREF_MAX.flatten(), [1]))


# NEURAL-LANDER FUNCTIONS
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        if not x.is_cuda:
            self.cpu()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def read_weight(filename):
    model_weight = torch.load(filename, map_location=torch.device("cpu"))
    model = Network().double()
    model.load_state_dict(model_weight)
    model = model.float()
    # .cuda()
    return model


Fa_model = read_weight("model/Fa_net_12_3_full_Lip16.pth")


def Fa_func(x: torch.Tensor):
    z = x[:, 2]
    vx = x[:, 3]
    vy = x[:, 4]
    vz = x[:, 5]

    if next(Fa_model.parameters()).device != z.device:
        Fa_model.to(z.device)
    # use prediction from NN as ground truth
    n = z.shape[0]
    state = torch.zeros((n, 12))
    state[:, 0] = z + drone_height
    state[:, 1] = vx  # velocity
    state[:, 2] = vy  # velocity
    state[:, 3] = vz  # velocity
    state[:, 7] = 1.0
    state[:, 8:12] = 6508.0 / 8000

    with torch.no_grad():
        Fa = Fa_model(state) * torch.tensor([30.0, 15.0, 10.0])

    return Fa


def Fa_func_np(x):
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    x = torch.tensor(x).float().view(1, -1)
    Fa = Fa_func(x).numpy()
    return Fa


class NeuralLanderEnv(gym.Env):
    def __init__(self, sigma: float = 0.0):
        super(NeuralLanderEnv, self).__init__()
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """
        self.num_dim_x = 6
        self.num_dim_control = 3
        self.pos_dimension = 3

        self.tracking_scaler = 1.0
        self.control_scaler = 0.0

        self.time_bound = 6.0
        self.dt = 0.03
        self.max_episode_len = int(self.time_bound / self.dt)
        self.episode_len = int(self.time_bound / self.dt)
        self.t = np.arange(0, self.time_bound, self.dt)

        self.state_weights = state_weights
        self.sigma = sigma
        self.d_up = 3 * sigma

        self.effective_indices = np.arange(2, 6)
        self.Bbot_func = None

        self.use_learned_dynamics = False

        # initialize the ref trajectory
        self.x_0, self.xref, self.uref, self.episode_len = self.system_reset()
        self.init_tracking_error = np.linalg.norm(self.x_0 - self.xref[0], ord=2)

        self.disturbance_duration = 0.0
        self.current_disturbance = np.zeros(self.num_dim_x)

        self.observation_space = spaces.Box(
            low=STATE_MIN.flatten(), high=STATE_MAX.flatten(), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=UREF_MIN.flatten(), high=UREF_MAX.flatten(), dtype=np.float64
        )

    def replace_dynamics(self, dynamics_model: nn.Module):
        print("[INFO] The environment is now using learned dynamics for transition.")
        self.learned_dynamics_model = deepcopy(dynamics_model).cpu()
        self.learned_dynamics_model.device = torch.device("cpu")
        self.use_learned_dynamics = True

    def f_func(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            n = x.shape[0]

            Fa = Fa_func(x)
            x, y, z, vx, vy, vz = [x[:, i] for i in range(self.num_dim_x)]

            f = torch.zeros((n, self.num_dim_x))
            f[:, 0] = vx
            f[:, 1] = vy
            f[:, 2] = vz
            f[:, 3] = Fa[:, 0] / mass
            f[:, 4] = Fa[:, 1] / mass
            f[:, 5] = Fa[:, 2] / mass - g
        else:
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            n = x.shape[0]

            Fa = Fa_func_np(x)
            x, y, z, vx, vy, vz = [x[:, i] for i in range(self.num_dim_x)]

            f = np.zeros((n, self.num_dim_x))
            f[:, 0] = vx
            f[:, 1] = vy
            f[:, 2] = vz
            f[:, 3] = Fa[:, 0] / mass
            f[:, 4] = Fa[:, 1] / mass
            f[:, 5] = Fa[:, 2] / mass - g

        return f.squeeze()

    def B_func(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            n = x.shape[0]

            B = torch.zeros((n, self.num_dim_x, self.num_dim_control))

            B[:, 3, 0] = 1
            B[:, 4, 1] = 1
            B[:, 5, 2] = 1
        else:
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            n = x.shape[0]

            B = np.zeros((n, self.num_dim_x, self.num_dim_control))

            B[:, 3, 0] = 1
            B[:, 4, 1] = 1
            B[:, 5, 2] = 1

        return B.squeeze()

    def B_null(self, x: torch.Tensor | np.ndarray):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            n = x.shape[0]

            Bbot = torch.cat(
                (
                    torch.eye(
                        self.num_dim_x - self.num_dim_control,
                        self.num_dim_x - self.num_dim_control,
                    ),
                    torch.zeros(
                        (self.num_dim_control, self.num_dim_x - self.num_dim_control)
                    ),
                ),
                dim=0,
            )
            return Bbot.repeat(n, 1, 1).squeeze()
        else:
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            n = x.shape[0]

            Bbot = np.concatenate(
                (
                    np.eye(
                        self.num_dim_x - self.num_dim_control,
                        self.num_dim_x - self.num_dim_control,
                    ),
                    np.zeros(
                        (self.num_dim_control, self.num_dim_x - self.num_dim_control)
                    ),
                ),
                axis=0,
            )
            return np.repeat(Bbot[np.newaxis, :, :], n, axis=0).squeeze()

    def get_f_and_B(self, x: torch.Tensor | np.ndarray):
        if self.use_learned_dynamics:
            with torch.no_grad():
                f_x, B_x, Bbot_x = self.learned_dynamics_model(x)
            return (
                f_x.cpu().squeeze(0).numpy(),
                B_x.cpu().squeeze(0).numpy(),
                Bbot_x.cpu().squeeze(0).numpy(),
            )
        else:
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
        Fa = Fa_func_np(xref_0.reshape(-1)).reshape(-1)

        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (
            0.5 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))
        ).tolist()

        def sample_controls():
            uref = np.array([0, 0, g]) - Fa / mass  # ref
            for freq, weight in zip(freqs, weights):
                uref += np.array(
                    [
                        weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                        weight[1] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                        weight[2] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    ]
                )
            uref = np.clip(uref, 0.75 * UREF_MIN.flatten(), 0.75 * UREF_MAX.flatten())
            return uref

        xref_list, uref_list = [xref_0], []
        for i, _t in enumerate(self.t):
            uref = sample_controls()

            xref_t = xref_list[-1].copy()
            f_xref, B_xref = self.f_func(xref_t), self.B_func(xref_t)
            xref_dot = f_xref + np.matmul(B_xref, uref[:, np.newaxis]).squeeze()

            xref_t = xref_t + self.dt * xref_dot

            termination = np.any(
                xref_t[: self.pos_dimension] <= X_MIN.flatten()[: self.pos_dimension]
            ) or np.any(
                xref_t[: self.pos_dimension] >= X_MAX.flatten()[: self.pos_dimension]
            )

            xref_t = np.clip(xref_t, X_MIN.flatten(), X_MAX.flatten())

            xref_list.append(xref_t)
            uref_list.append(uref)

            if termination:
                break

        return (
            x_0,
            np.array(xref_list),
            np.array(uref_list),
            i,
        )

    def dynamic_fn(self, action):
        f_x, B_x, _ = self.get_f_and_B(self.x_t)

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

        # self.state = np.concatenate((self.x_t, self.xref[self.time_steps]))
        self.state = np.concatenate(
            (
                self.x_t,
                self.xref[self.time_steps],
                self.uref[self.time_steps],
                [self.time_steps / self.max_episode_len],
            )
        )
        self.time_steps += 1

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
            (
                self.x_t,
                self.xref[self.time_steps],
                self.uref[self.time_steps],
                [self.time_steps / self.max_episode_len],
            )
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
            xref = (X_MAX - X_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_x
            ) + X_MIN.flatten()
            uref = (UREF_MAX - UREF_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_control
            ) + UREF_MIN.flatten()
            xe = (XE_MAX - XE_MIN).flatten() * np.random.rand(
                buffer_size, self.num_dim_x
            ) + XE_MIN.flatten()

            # Compose states
            x = xe + xref
            x = np.clip(x, X_MIN.flatten(), X_MAX.flatten())

            # Store
            c3m_data["x"] = x.astype(np.float32)
            c3m_data["xref"] = xref.astype(np.float32)
            c3m_data["uref"] = uref.astype(np.float32)

            # Check for NaNs
            if np.any(np.isnan(c3m_data["x"])):
                print("NaN values found in x")

            return c3m_data

        else:
            dynamics_data = dict(
                x=np.full(((buffer_size, self.num_dim_x)), np.nan, dtype=np.float32),
                u=np.full(
                    ((buffer_size, self.num_dim_control)), np.nan, dtype=np.float32
                ),
                x_dot=np.full((buffer_size, self.num_dim_x), np.nan, dtype=np.float32),
            )

            # === DATA FOR DYNAMICS LEARNING === #
            n_control_per_x = 3
            batch_size = ceil(buffer_size / n_control_per_x)

            # sample_mode = "Gaussian"
            sample_mode = "Uniform"
            # sample_mode = "Sampling"
            if sample_mode == "Gaussian":
                # Compute mean and std for Gaussian distribution
                x_mean = (X_MAX.flatten() + X_MIN.flatten()) / 2.0
                x_std = (X_MAX.flatten() - X_MIN.flatten()) / 6.0  # 3σ covers range

                u_mean = (UREF_MAX.flatten() + UREF_MIN.flatten()) / 2.0
                u_std = (UREF_MAX.flatten() - UREF_MIN.flatten()) / 6.0

                # Sample Gaussian-distributed data
                x = np.random.normal(
                    loc=x_mean,
                    scale=x_std,
                    size=(batch_size, len(x_mean)),
                )

                u = np.random.normal(
                    loc=u_mean,
                    scale=u_std,
                    size=(batch_size, len(u_mean)),
                )

                # Step 1: Repeat x n_control_per_x times along axis 0
                x = np.concatenate([x] * n_control_per_x, axis=0)

                # Step 2: Shuffle u independently n_control_per_x times and stack
                u = np.concatenate(
                    [u[np.random.permutation(len(u))] for _ in range(n_control_per_x)],
                    axis=0,
                )

                f_x, B_x, _ = self.get_f_and_B(x)
                x_dot = f_x + np.matmul(B_x, u[:, :, np.newaxis]).squeeze()

                dynamics_data["x"] = x[:buffer_size].astype(np.float32)
                dynamics_data["u"] = u[:buffer_size].astype(np.float32)
                dynamics_data["x_dot"] = x_dot[:buffer_size].astype(np.float32)

            elif sample_mode == "Uniform":
                # Original sampling
                x = np.random.uniform(
                    low=X_MIN.flatten(),
                    high=X_MAX.flatten(),
                    size=(batch_size, len(X_MAX.flatten())),
                )
                u = np.random.uniform(
                    low=UREF_MIN.flatten(),
                    high=UREF_MAX.flatten(),
                    size=(batch_size, len(UREF_MAX.flatten())),
                )

                # Step 1: Repeat x n_control_per_x times along axis 0
                x = np.concatenate([x] * n_control_per_x, axis=0)

                # Step 2: Shuffle u independently n_control_per_x times and stack
                u = np.concatenate(
                    [u[np.random.permutation(len(u))] for _ in range(n_control_per_x)],
                    axis=0,
                )

                f_x, B_x, _ = self.get_f_and_B(x)
                x_dot = f_x + np.matmul(B_x, u[:, :, np.newaxis]).squeeze()

                dynamics_data["x"] = x[:buffer_size].astype(np.float32)
                dynamics_data["u"] = u[:buffer_size].astype(np.float32)
                dynamics_data["x_dot"] = x_dot[:buffer_size].astype(np.float32)
            else:
                current_time = 0
                while current_time < buffer_size:
                    xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (
                        X_INIT_MAX - X_INIT_MIN
                    )
                    xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (
                        XE_INIT_MAX - XE_INIT_MIN
                    )
                    x_0 = xref_0 + xe_0

                    Fa = Fa_func_np(xref_0.reshape(-1)).reshape(-1)

                    freqs = list(range(1, 11))
                    weights = np.random.randn(len(freqs), len(UREF_MIN))
                    weights = (
                        0.5 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))
                    ).tolist()

                    def sample_controls():
                        u = np.array([0, 0, g]) - Fa / mass  # ref
                        for freq, weight in zip(freqs, weights):
                            u += np.array(
                                [
                                    weight[0]
                                    * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                                    weight[1]
                                    * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                                    weight[2]
                                    * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                                ]
                            )
                        # add gaussian noise
                        u += np.random.normal(0, np.abs(0.1 * u), size=u.shape)
                        u = np.clip(u, UREF_MIN.flatten(), UREF_MAX.flatten())
                        return u

                    x_list = [x_0]
                    for i, _t in enumerate(self.t):
                        u = sample_controls()

                        x_t = x_list[-1].copy()
                        f_x, B_x = self.f_func(x_t), self.B_func(x_t)
                        x_dot = f_x + np.matmul(B_x, u[:, np.newaxis]).squeeze()

                        ### LOGGING ###
                        try:
                            dynamics_data["x"][current_time + i] = x_list[-1]
                            dynamics_data["u"][current_time + i] = u
                            dynamics_data["x_dot"][current_time + i] = x_dot
                        except:
                            break

                        x_t = x_t + self.dt * x_dot

                        termination = np.any(
                            x_t[: self.pos_dimension]
                            <= X_MIN.flatten()[: self.pos_dimension]
                        ) or np.any(
                            x_t[: self.pos_dimension]
                            >= X_MAX.flatten()[: self.pos_dimension]
                        )

                        x_t = np.clip(x_t, X_MIN.flatten(), X_MAX.flatten())

                        x_list.append(x_t)

                        if termination:
                            break

                    current_time += i + 1

            return dynamics_data
