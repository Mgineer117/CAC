from abc import ABC, abstractmethod
from copy import deepcopy
from math import ceil, floor
from time import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces


class BaseEnv(gym.Env):
    """Base class for all environments."""

    def __init__(self, env_config: dict):
        super(BaseEnv, self).__init__()

        # X bounds
        self.X_MIN = env_config["x_min"]
        self.X_MAX = env_config["x_max"]

        # Initial reference state bounds
        self.XREF_INIT_MIN = env_config["xref_init_min"]
        self.XREF_INIT_MAX = env_config["xref_init_max"]

        # Initial reference state perturbation bounds
        self.XE_INIT_MIN = env_config["xe_init_min"]
        self.XE_INIT_MAX = env_config["xe_init_max"]

        # Reference state perturbation bounds for c3m
        self.XE_MIN = env_config["xe_min"]
        self.XE_MAX = env_config["xe_max"]

        # Reference control bounds
        self.UREF_MIN = env_config["uref_min"]
        self.UREF_MAX = env_config["uref_max"]

        # overall state bounds
        self.STATE_MIN = np.concatenate(
            (self.X_MIN.flatten(), self.X_MIN.flatten(), self.UREF_MIN.flatten(), [0])
        )
        self.STATE_MAX = np.concatenate(
            (self.X_MAX.flatten(), self.X_MAX.flatten(), self.UREF_MAX.flatten(), [1])
        )

        # gymnasium spaces
        self.observation_space = spaces.Box(
            low=self.STATE_MIN.flatten(),
            high=self.STATE_MAX.flatten(),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=self.UREF_MIN.flatten(),
            high=self.UREF_MAX.flatten(),
            dtype=np.float64,
        )

        # environment parameters
        self.num_dim_x = env_config["num_dim_x"]  # x, y, theta, v
        self.num_dim_control = env_config[
            "num_dim_control"
        ]  # u1 (angular acc), u2 (linear acc)
        self.pos_dimension = env_config["pos_dimension"]

        self.time_bound = env_config["time_bound"]
        self.dt = env_config["dt"]
        self.max_episode_len = int(self.time_bound / self.dt)
        self.episode_len = int(self.time_bound / self.dt)
        self.t = np.arange(0, self.time_bound, self.dt)

        # dynamics parameters
        self.tracking_scaler = env_config["q"]
        self.control_scaler = env_config["r"]

        # etc parameters
        self.use_learned_dynamics = False
        self.sample_mode = env_config["sample_mode"]
        self.reward_mode = env_config["reward_mode"]

        # reset
        self.reset()

    def reset(self, seed=None, options: dict | None = None):
        """Resets the environment to an initial state and returns an initial observation."""
        super().reset(seed=seed)
        self.time_steps = 0

        # Initialize the state
        if options is None:
            # Default reset behavior
            self.x_0, self.xref, self.uref, self.episode_len = self.system_reset()
        else:
            # Custom reset behavior that keeps reference trajectory but changes initial state
            if options.get("replace_x_0", True):
                _, xe_0, _ = self.define_initial_state()
                self.x_0 = self.xref[0] + xe_0

        self.init_tracking_error = np.linalg.norm(self.x_0 - self.xref[0], ord=2)

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

    def step(self, u):
        """Run one timestep of the environment's dynamics."""
        # Construct u and apply u clipping
        u = self.uref[self.time_steps] + u
        # Get reward
        reward, infos = self.get_rewards(u)
        # Clip control to bounds
        u = np.clip(u, self.UREF_MIN.flatten(), self.UREF_MAX.flatten())

        # Get next state
        self.x_t, termination, truncation = self.get_transition(self.x_t, u)
        # Clip state to bounds
        self.x_t = np.clip(self.x_t, self.X_MIN.flatten(), self.X_MAX.flatten())

        # Construct observation
        self.state = np.concatenate(
            (
                self.x_t,
                self.xref[self.time_steps],
                self.uref[self.time_steps],
                [self.time_steps / self.max_episode_len],
            )
        )

        # Update time step
        self.time_steps += 1

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

    def replace_dynamics(self, dynamics_model: nn.Module):
        print("[INFO] The environment is now using learned dynamics for transition.")
        self.learned_dynamics_model = deepcopy(dynamics_model).cpu()
        self.learned_dynamics_model.device = torch.device("cpu")
        self.use_learned_dynamics = True

    @abstractmethod
    def _f_logic(self, x: torch.Tensor | np.ndarray, lib):
        """Logic for calculating f(x) given a library (torch or numpy)."""
        pass

    @abstractmethod
    def _B_logic(self, x: torch.Tensor | np.ndarray, lib):
        """Logic for calculating B(x) given a library (torch or numpy)."""
        pass

    def _B_null_logic(self, x, n, lib):
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

    def f_func(self, x: torch.Tensor | np.ndarray):
        """Calculates the drift dynamics f(x) for torch or numpy."""
        if isinstance(x, torch.Tensor):
            lib = torch
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            result = self._f_logic(x, lib)
        else:
            lib = np
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            result = self._f_logic(x, lib)

        try:
            return result.squeeze(0)
        except:
            return result

    def B_func(self, x: torch.Tensor | np.ndarray):
        """Calculates the control matrix B(x) for torch or numpy."""
        if isinstance(x, torch.Tensor):
            lib = torch
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            result = self._B_logic(x, lib)
        else:
            lib = np
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            result = self._B_logic(x, lib)

        try:
            return result.squeeze(0)
        except:
            return result

    def B_null(self, x: torch.Tensor | np.ndarray):
        """Calculates the null space of B for torch or numpy."""

        # Check type and get the batch size 'n'
        if isinstance(x, torch.Tensor):
            lib = torch
            n = 1 if len(x.shape) == 1 else x.shape[0]
            result = self._B_null_logic(x, n, lib)
        else:
            lib = np
            n = 1 if len(x.shape) == 1 else x.shape[0]
            result = self._B_null_logic(x, n, lib)

        # .squeeze() removes the batch dimension if the input was 1D
        try:
            return result.squeeze(0)
        except:
            return result

    def get_f_and_B(self, x: torch.Tensor | np.ndarray):
        """Get f(x), B(x), and B_null(x) using either learned dynamics or analytical functions."""
        if self.use_learned_dynamics:
            with torch.no_grad():
                f_x, B_x, Bbot_x = self.learned_dynamics_model(x)
            return (
                f_x.cpu().squeeze(0).numpy(),
                B_x.cpu().squeeze(0).numpy(),
                Bbot_x.cpu().squeeze(0).numpy(),
            )
        else:
            return self.f_func(x), self.B_func(x), self.B_null(x)

    def get_dynamics(self, x: np.ndarray, u: np.ndarray):
        """Compute the dynamics x_dot given current state x and action u."""
        f_x, B_x, _ = self.get_f_and_B(x)
        x_dot = f_x + np.matmul(B_x, u[..., np.newaxis]).squeeze()

        return x_dot

    def get_transition(self, x: np.ndarray, u: np.ndarray):
        """Compute the next state given current state x and action u."""
        x_dot = self.get_dynamics(x, u)
        next_x = x + self.dt * x_dot

        next_x, termination, truncation = self.post_process_state(next_x)

        return next_x, termination, truncation

    def post_process_state(self, x: np.ndarray):
        """Post-process the state if needed (e.g., wrapping angles)."""
        termination = np.any(
            x[: self.pos_dimension] <= self.X_MIN.flatten()[: self.pos_dimension]
        ) or np.any(
            x[: self.pos_dimension] >= self.X_MAX.flatten()[: self.pos_dimension]
        )
        truncation = self.time_steps == self.episode_len - 1

        x = np.clip(x, self.X_MIN.flatten(), self.X_MAX.flatten())
        return x, termination, truncation

    def define_initial_state(self):
        """Define the initial state of the environment."""
        xref_0 = self.XREF_INIT_MIN + np.random.rand(len(self.XREF_INIT_MIN)) * (
            self.XREF_INIT_MAX - self.XREF_INIT_MIN
        )
        xe_0 = self.XE_INIT_MIN + np.random.rand(len(self.XE_INIT_MIN)) * (
            self.XE_INIT_MAX - self.XE_INIT_MIN
        )
        x_0 = xref_0 + xe_0

        return xref_0, xe_0, x_0

    @abstractmethod
    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise):
        """Sample reference controls based on frequencies and weights."""
        pass

    def system_reset(self):
        """Resets the system to an initial state and generates a reference trajectory."""
        xref_0, xe_0, x_0 = self.define_initial_state()

        # Generate reference trajectory
        freqs = list(range(1, 11))
        weights = np.random.randn(len(freqs), len(self.UREF_MIN))
        weights = (weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()

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

    def get_rewards(self, u):
        error = self.x_t - self.xref[self.time_steps]

        tracking_error = np.linalg.norm(
            error,
            ord=2,
        )
        control_effort = np.linalg.norm(u, ord=2)

        tracking_reward = -self.tracking_scaler * tracking_error
        control_reward = -self.control_scaler * control_effort

        if self.reward_mode == "inverse":
            tracking_reward = 1 / (1 + tracking_reward)
            control_reward = 1 / (1 + control_reward)

        reward = (0.5 * tracking_reward) + (0.5 * control_reward)

        return reward, {
            "tracking_error": tracking_error,
            "control_effort": control_effort,
        }

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
            dynamics_data = dict(
                x=np.full(
                    ((buffer_size + self.max_episode_len, self.num_dim_x)),
                    np.nan,
                    dtype=np.float32,
                ),
                u=np.full(
                    ((buffer_size + self.max_episode_len, self.num_dim_control)),
                    np.nan,
                    dtype=np.float32,
                ),
                x_dot=np.full(
                    (buffer_size + self.max_episode_len, self.num_dim_x),
                    np.nan,
                    dtype=np.float32,
                ),
            )

            # === DATA FOR DYNAMICS LEARNING === #
            n_control_per_x = 3
            batch_size = ceil(buffer_size / n_control_per_x)

            if self.sample_mode == "Gaussian":
                # Compute mean and std for Gaussian distribution
                x_mean = (self.X_MAX.flatten() + self.X_MIN.flatten()) / 2.0
                x_std = (
                    self.X_MAX.flatten() - self.X_MIN.flatten()
                ) / 6.0  # 3σ covers range

                u_mean = (self.UREF_MAX.flatten() + self.UREF_MIN.flatten()) / 2.0
                u_std = (self.UREF_MAX.flatten() - self.UREF_MIN.flatten()) / 6.0

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

                x_dot = self.get_dynamics(x, u)

                dynamics_data["x"][:buffer_size] = x[:buffer_size].astype(np.float32)
                dynamics_data["u"][:buffer_size] = u[:buffer_size].astype(np.float32)
                dynamics_data["x_dot"][:buffer_size] = x_dot[:buffer_size].astype(
                    np.float32
                )

            elif self.sample_mode == "Uniform":
                # Original sampling
                x = np.random.uniform(
                    low=self.X_MIN.flatten(),
                    high=self.X_MAX.flatten(),
                    size=(batch_size, len(self.X_MAX.flatten())),
                )
                u = np.random.uniform(
                    low=self.UREF_MIN.flatten(),
                    high=self.UREF_MAX.flatten(),
                    size=(batch_size, len(self.UREF_MAX.flatten())),
                )

                # Step 1: Repeat x n_control_per_x times along axis 0
                x = np.concatenate([x] * n_control_per_x, axis=0)

                # Step 2: Shuffle u independently n_control_per_x times and stack
                u = np.concatenate(
                    [u[np.random.permutation(len(u))] for _ in range(n_control_per_x)],
                    axis=0,
                )

                x_dot = self.get_dynamics(x, u)

                dynamics_data["x"][:buffer_size] = x[:buffer_size].astype(np.float32)
                dynamics_data["u"][:buffer_size] = u[:buffer_size].astype(np.float32)
                dynamics_data["x_dot"][:buffer_size] = x_dot[:buffer_size].astype(
                    np.float32
                )
            else:
                current_time = 0
                while current_time < buffer_size:
                    xref_0, xe_0, x_0 = self.define_initial_state()

                    freqs = list(range(1, 11))
                    weights = np.random.randn(len(freqs), len(self.UREF_MIN))
                    weights = (
                        weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))
                    ).tolist()

                    x_list = [x_0]
                    for i, _t in enumerate(self.t):
                        x_t = x_list[-1].copy()
                        u_t = self.sample_reference_controls(
                            freqs, weights, _t, {"xref_0": xref_0}, add_noise=True
                        )
                        x_dot = self.get_dynamics(x_t, u_t)
                        x_t, term, _ = self.get_transition(x_t, u_t)

                        ### LOGGING ###
                        dynamics_data["x"][current_time + i] = x_t
                        dynamics_data["u"][current_time + i] = u_t
                        dynamics_data["x_dot"][current_time + i] = x_dot

                        x_t = np.clip(x_t, self.X_MIN.flatten(), self.X_MAX.flatten())

                        x_list.append(x_t)

                        # here trunc is not necessary since we use for loops.
                        if term:
                            break

                    current_time += i + 1

                dynamics_data["x"] = dynamics_data["x"][:buffer_size]
                dynamics_data["u"] = dynamics_data["u"][:buffer_size]
                dynamics_data["x_dot"] = dynamics_data["x_dot"][:buffer_size]

            return dynamics_data
