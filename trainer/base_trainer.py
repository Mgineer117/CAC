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
    "0": "#4e79a7",  # Blue
    "1": "#f28e2c",  # Orange
    "2": "#e15759",  # Red
    "3": "#76b7b2",  # Teal
    "4": "#59a14f",  # Green
    "5": "#edc949",  # Yellow
    "6": "#af7aa1",  # Purple
    "7": "#ff9da7",  # Pink
    "8": "#9c755f",  # Brown
    "9": "#bab0ab",  # Grey
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
        ep_buffers = []

        # find mean and CI of data with tqdm that disappears afterward
        for i in tqdm(range(self.eval_num), desc="Evaluating", leave=False):
            track_traj, ref_traj, error_traj, ep_buffer = [], [], [], []
            # find mean of data
            for j in range(self.eval_episodes):
                # Env initialization
                options = None if j == 0 else {"replace_x_0": True}
                obs, infos = self.eval_env.reset(seed=self.seed, options=options)

                # Episode variables
                ep_return, ep_ctrl_effort, ep_inf_time = 0, 0, 0
                ep_track_traj, ep_error_traj = [], []

                # Episode rollout
                for t in range(1, self.eval_env.episode_len + 1):
                    with torch.no_grad():
                        t0 = time.time()
                        a, _ = self.policy(obs, deterministic=True)
                        t1 = time.time()
                        a = (
                            a.cpu().numpy().squeeze(0)
                            if a.shape[-1] > 1
                            else [a.item()]
                        )

                    obs, rew, term, trunc, infos = self.eval_env.step(a)
                    done = term or trunc

                    # === Logging === #
                    ep_return += self.policy.gamma**t * rew
                    ep_ctrl_effort += infos["control_effort"]
                    ep_inf_time += t1 - t0

                    ep_track_traj.append(infos["x"][:dimension])
                    ep_error_traj.append(infos["relative_tracking_error"])
                    if j == 0:
                        ref_traj.append(self.eval_env.xref[t, :dimension])

                    # === Termination logic === #
                    if done:
                        auc = np.trapezoid(ep_error_traj, dx=self.eval_env.dt)

                        ep_buffer.append(
                            {
                                "return": ep_return,
                                "avg_ctrl_effort": ep_ctrl_effort / t,
                                "avg_inf_time": ep_inf_time / t,
                                "mauc": auc * (len(ep_track_traj) / t),
                                "episode_len": t + 1,
                            }
                        )
                        track_traj.append(ep_track_traj)
                        error_traj.append(ep_error_traj)

                        break

            # === ref traj level logging === #
            rew_list = [ep_info["return"] for ep_info in ep_buffer]
            ctr_list = [ep_info["avg_ctrl_effort"] for ep_info in ep_buffer]
            inf_list = [ep_info["avg_inf_time"] for ep_info in ep_buffer]
            mauc_list = [ep_info["mauc"] for ep_info in ep_buffer]

            ret_mean, _ = self.mean_confidence_interval(rew_list)
            inf_mean, _ = self.mean_confidence_interval(inf_list)
            mauc_mean, _ = self.mean_confidence_interval(mauc_list)
            ctrl_mean, _ = self.mean_confidence_interval(ctr_list)

            C, lbd = self.compute_contraction_rate(error_traj)

            if i == 0:
                fig = self.plot_trajectories(track_traj, error_traj, dimension)

            #
            ep_buffers.append(
                {
                    "return": ret_mean,
                    "avg_ctrl_effort": ctrl_mean,
                    "avg_inf_time": inf_mean,
                    "mauc": mauc_mean,
                    "overshoot": C,
                    "contraction_rate": lbd / C,
                }
            )

        # === eval num level logging === #
        rew_list = [ep_info["return"] for ep_info in ep_buffers]
        ctr_list = [ep_info["avg_ctrl_effort"] for ep_info in ep_buffers]
        inf_list = [ep_info["avg_inf_time"] for ep_info in ep_buffers]
        mauc_list = [ep_info["mauc"] for ep_info in ep_buffers]
        overshoot_list = [ep_info["overshoot"] for ep_info in ep_buffers]
        lbd_list = [ep_info["contraction_rate"] for ep_info in ep_buffers]

        ret_mean, ret_ci = self.mean_confidence_interval(rew_list)
        inf_mean, inf_ci = self.mean_confidence_interval(inf_list)
        mauc_mean, mauc_ci = self.mean_confidence_interval(mauc_list)
        ctrl_mean, ctrl_ci = self.mean_confidence_interval(ctr_list)
        overshoot_mean, overshoot_ci = self.mean_confidence_interval(overshoot_list)
        lbd_mean, lbd_ci = self.mean_confidence_interval(lbd_list)

        eval_dict = {
            f"eval/return_mean": ret_mean,
            f"eval/return_ci(95)": ret_ci,
            f"eval/inf_mean": inf_mean,
            f"eval/inf_ci(95)": inf_ci,
            f"eval/mauc_mean": mauc_mean,
            f"eval/mauc_ci(95)": mauc_ci,
            f"eval/control_effort_mean": ctrl_mean,
            f"eval/control_effort_ci(95)": ctrl_ci,
            f"eval/overshoot": overshoot_mean,
            f"eval/overshoot_ci(95)": overshoot_ci,
            f"eval/contraction_rate": lbd_mean,
            f"eval/contraction_rate_ci(95)": lbd_ci,
        }

        supp_dict = {f"eval/path_tracking_result": fig}

        return eval_dict, supp_dict

    def compute_contraction_rate(self, error_trajectories: list[np.ndarray]):
        C = 1.0
        for err in error_trajectories:
            max_err = np.max(err)
            if max_err > C:
                C = max_err

        lbds = []
        for err in error_trajectories:
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

        return C, lbd

    def mean_confidence_interval(self, data, confidence=0.95):
        n = len(data)
        data = np.array(data)
        mean = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(n)  # standard error
        h = 1.96 * sem  # margin of error for 95% CI
        return mean, h

    def plot_trajectories(
        self,
        trajectories: list[np.ndarray],
        error_trajectories: list[np.ndarray],
        dimension: int,
    ):
        assert dimension in [1, 2, 3], "Dimension must be 1, 2, or 3."

        # Set subplot parameters based on dimension
        if dimension == 3:
            fig = plt.figure(figsize=(14, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2)  # 2D subplot
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        if dimension in [2, 3]:
            # Dynamically create the coordinate list and plot the reference trajectory
            coords = [self.eval_env.xref[:, i] for i in range(dimension)]
        elif dimension == 1:
            # for one dimensional env (e.g., Segway) we plot x vs time
            coords = [np.arange(len(self.eval_env.xref)), self.eval_env.xref[:, 0]]

        first_point = [c[0] for c in coords]
        ax1.scatter(
            *first_point,
            marker="*",
            # alpha=0.7,
            c="black",
            s=80.0,
        )
        ax1.plot(*coords, linewidth=2.0, linestyle="--", c="black", label="Reference")

        for num_episodes, trajectory in enumerate(trajectories):
            trajectory = np.array(trajectory)
            if dimension in [2, 3]:
                coords = [trajectory[:, i] for i in range(dimension)]
            else:
                coords = [np.arange(len(trajectory)), trajectory[:, 0]]
            first_point = [c[0] for c in coords]
            ax1.scatter(
                *first_point,
                marker="*",
                alpha=0.9,
                c=COLORS[str(num_episodes)],
                s=80.0,
            )
            ax1.plot(
                *coords,
                linestyle="-",
                alpha=0.9,
                c=COLORS[str(num_episodes)],
                label=str(num_episodes),
            )

        # Optional: Add axis labels
        if dimension in [2, 3]:
            ax1.set_xlabel("X", fontsize=16)
            ax1.set_ylabel("Y", fontsize=16)
            if dimension == 3:
                ax1.set_zlabel("Z", fontsize=16)
                # Set a nice viewing angle for 3D
                ax1.view_init(elev=25, azim=45)
        else:
            ax1.set_xlabel("Time Steps", fontsize=16)
            ax1.set_ylabel("Position", fontsize=16)

        ax1.set_title("Path Tracking Results", fontsize=18)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # calculate the mean and std of the traj norm error to make plot
        i = 0
        for traj in error_trajectories:
            ax2.plot(
                traj,
                # alpha=0.7,
                c=COLORS[str(i)],
            )
            i += 1
        ax2.set_xlabel("Time Steps", fontsize=16)
        ax2.set_ylabel(r"$||x(t)-x^*(t)||_2 / ||x(0) - x^*(0)||_2$", fontsize=16)

        ax2.set_title("Normalized Tracking Error", fontsize=18)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.close()

        return fig

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
