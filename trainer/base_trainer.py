import os
import time
from abc import abstractmethod
from collections import deque
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.base import Base

COLORS = {
    "0": "magenta",
    "1": "red",
    "2": "blue",
    "3": "green",
    "4": "yellow",
    "5": "orange",
    "6": "purple",
    "7": "pink",
    "8": "brown",
    "9": "grey",
}


class BaseTrainer:
    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        policy: Base,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_epochs: int = 0,
        epochs: int = 10000,
        log_interval: int = 2,
        eval_num: int = 10,
        eval_episodes: int = 10,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        self.logger = logger
        self.writer = writer

        # training parameters
        self.init_epochs = init_epochs
        self.epochs = epochs

        self.log_interval = log_interval
        self.eval_interval = int(epochs / self.log_interval)

        # initialize the essential training components
        self.last_min_auc_mean = 1e10

        self.eval_num = eval_num
        self.eval_episodes = eval_episodes
        self.seed = seed

    @abstractmethod
    def train(self) -> dict[str, float]:
        pass

    def evaluate(self):
        """
        Given one ref, show tracking performance
        """
        dimension = self.eval_env.pos_dimension

        # Set subplot parameters based on dimension
        if dimension == 3:
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2)  # 2D subplot
        elif dimension == 2:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

        # Dynamically create the coordinate list and plot the reference trajectory
        coords = [self.eval_env.xref[:, i] for i in range(dimension)]

        first_point = [c[0] for c in coords]
        ax1.scatter(
            *first_point,
            marker="*",
            alpha=0.7,
            c="black",
            s=80.0,
        )
        ax1.plot(*coords, linestyle="--", c="black", label="Reference")

        # error_norm_trajs = []
        error_trajs = []
        ep_buffer = []
        for num_episodes in range(self.eval_episodes):
            ep_reward, ep_tracking_error, ep_control_effort, ep_inference_time = (
                0,
                0,
                0,
                0,
            )
            normalized_tracking_error = [1.0]

            # Env initialization
            options = {"replace_x_0": True}
            obs, infos = self.eval_env.reset(seed=self.seed, options=options)

            tref_trajectory = [self.eval_env.time_steps]
            trajectory = [infos["x"][:dimension]]
            for t in range(1, self.eval_env.episode_len + 1):
                with torch.no_grad():
                    t0 = time.time()
                    a, _ = self.policy(obs, deterministic=True)
                    t1 = time.time()
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                next_obs, rew, term, trunc, infos = self.eval_env.step(a)

                tref_trajectory.append(self.eval_env.time_steps)
                trajectory.append(infos["x"][:dimension])  # Store trajectory point
                normalized_tracking_error.append(infos["relative_tracking_error"])

                done = term or trunc

                obs = next_obs
                ep_reward += rew
                ep_inference_time += t1 - t0
                ep_tracking_error += infos["tracking_error"]
                ep_control_effort += infos["control_effort"]

                if done:
                    auc = np.trapezoid(normalized_tracking_error, dx=self.eval_env.dt)
                    ep_buffer.append(
                        {
                            "avg_reward": ep_reward / t,
                            "avg_inference_time": ep_inference_time / t,
                            "mauc": auc * (self.eval_env.episode_len / t),
                            "tracking_error": ep_tracking_error / t,
                            "control_effort": ep_control_effort / t,
                            "episode_len": t + 1,
                        }
                    )

                    # append the final point to ensure full trajectory is captured
                    for _ in range(t + 1, self.eval_env.episode_len):
                        normalized_tracking_error.append(
                            infos["relative_tracking_error"]
                        )

                    trajectory = np.array(trajectory)
                    coords = [trajectory[:, i] for i in range(dimension)]
                    first_point = [c[0] for c in coords]
                    ax1.scatter(
                        *first_point,
                        marker="*",
                        alpha=0.7,
                        c=COLORS[str(num_episodes)],
                        s=80.0,
                    )
                    ax1.plot(
                        *coords,
                        linestyle="-",
                        alpha=0.7,
                        c=COLORS[str(num_episodes)],
                        label=str(num_episodes),
                    )

                    # error_norm_trajs.append(error_norm_trajectory)
                    error_trajs.append(normalized_tracking_error)

                    break

        ### FIND OVERSHOOT AND CONVERGENCE RATE ###
        C = 1.0
        for err in error_trajs:
            max_err = np.max(err)
            if max_err > C:
                C = max_err

        lbds = []
        for err in error_trajs:
            lbd = float("inf")  # start high, then lower
            for i, xe in enumerate(err):
                val = (np.log(C) - np.log(xe)) / (self.eval_env.dt * (i + 1))
                lbd = min(lbd, val)
            # if not np.isfinite(lbd):
            #     lbd = 0.0  # or inf, depending on how you want to handle it
            lbds.append(max(0.0, lbd))  # enforce nonnegative
        # (1 - alpha)-quantile, here alpha=0.05
        # print(lbds)
        lbd = np.quantile(lbds, 0.95)

        # Optional: Add axis labels
        ax1.set_xlabel("X", labelpad=10)
        ax1.set_ylabel("Y", labelpad=10)
        if dimension == 3:
            ax1.set_zlabel("Z", labelpad=10)
            # Set a nice viewing angle for 3D
            ax1.view_init(elev=25, azim=45)

        # calculate the mean and std of the traj norm error to make plot
        i = 0
        for traj in zip(error_trajs):
            ax2.plot(
                traj,
                alpha=0.7,
                c=COLORS[str(i)],
            )
            i += 1
        ax2.set_xlabel("Time Steps", labelpad=10)
        ax2.set_ylabel(r"$||x(t)-x^*(t)||_2 / ||x(0) - x^*(0)||_2$", labelpad=10)

        plt.tight_layout()

        rew_list = [ep_info["avg_reward"] for ep_info in ep_buffer]
        inf_list = [ep_info["avg_inference_time"] for ep_info in ep_buffer]
        mauc_list = [ep_info["mauc"] for ep_info in ep_buffer]
        trk_list = [ep_info["tracking_error"] for ep_info in ep_buffer]
        ctr_list = [ep_info["control_effort"] for ep_info in ep_buffer]

        rew_mean, rew_interv = self.mean_confidence_interval(rew_list)
        inf_mean, inf_interv = self.mean_confidence_interval(inf_list)
        mauc_mean, mauc_interv = self.mean_confidence_interval(mauc_list)
        trk_mean, trk_interv = self.mean_confidence_interval(trk_list)
        ctr_mean, ctr_interv = self.mean_confidence_interval(ctr_list)

        eval_dict = {
            f"eval/rew_mean": rew_mean,
            f"eval/rew_std_(95)": rew_interv,
            f"eval/inf_mean": inf_mean,
            f"eval/inf_std_(95)": inf_interv,
            f"eval/mauc_mean": mauc_mean,
            f"eval/mauc_std_(95)": mauc_interv,
            f"eval/tracking_error_mean": trk_mean,
            f"eval/tracking_error_std_(95)": trk_interv,
            f"eval/control_effort_mean": ctr_mean,
            f"eval/control_effort_std_(95)": ctr_interv,
            f"eval/overshoot": C,
            f"eval/contraction_rate": lbd / C,
        }

        supp_dict = {f"eval/path_tracking_result(95)": fig}

        # Close the figure to free memory
        plt.close(fig)

        return eval_dict, supp_dict

    def mean_confidence_interval(self, data, confidence=0.95):
        n = len(data)
        data = np.array(data)
        mean = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(n)  # standard error
        h = 1.96 * sem  # margin of error for 95% CI
        return mean, h

    def plot_trajectories(
        self, trajectories: list[np.ndarray], title: str = "Trajectories"
    ):
        pass

    @abstractmethod
    def save_model(self, e):
        pass

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, supp_dict: dict, step: int):
        # supp_dict contains fig of plt
        for key, value in supp_dict.items():
            self.logger.write_images(step=step, image=value, logdir=key)

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values for each key
        sum_dict = {key: 0 for key in dict_list[0].keys()}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                sum_dict[key] += value

        # Calculate the average for each key
        avg_dict = {key: sum_val / len(dict_list) for key, sum_val in sum_dict.items()}

        return avg_dict
