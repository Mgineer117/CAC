import os
import time
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


class C3MTrainer:
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
        self.eval_interval = int((epochs + init_epochs) / self.log_interval)

        # initialize the essential training components
        self.last_min_auc_mean = 1e10
        self.last_min_auc_std = 1e10

        self.eval_num = eval_num
        self.eval_episodes = eval_episodes
        self.seed = seed

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_auc_mean = deque(maxlen=1)
        self.last_auc_std = deque(maxlen=1)

        # Train loop
        batch_size = 1024
        eval_idx = self.init_epochs // self.eval_interval
        self.policy.train()
        with tqdm(
            initial=self.init_epochs,
            total=(self.epochs + self.init_epochs),
            desc=f"{self.policy.name} Training (Epochs)",
        ) as pbar:
            while pbar.n < (self.epochs + self.init_epochs):
                step = pbar.n + 1  # + 1 to avoid zero division
                logging_step = (step - self.init_epochs) * batch_size + self.init_epochs

                loss_dict, update_time = self.policy.learn()

                # Calculate expected remaining time
                pbar.update(1)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.policy.name}/analytics/timesteps"] = logging_step
                loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time

                self.write_log(loss_dict, step=logging_step)

                #### EVALUATIONS ####
                if step >= self.eval_interval * eval_idx:
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict_list = []
                    for i in range(self.eval_num):
                        eval_dict, traj_plot = self.evaluate()
                        eval_dict_list.append(eval_dict)

                    eval_dict = self.average_dict_values(eval_dict_list)

                    # Manual logging
                    self.write_log(eval_dict, step=logging_step, eval_log=True)
                    self.write_image(
                        traj_plot,
                        step=logging_step,
                        logdir=f"eval",
                        name="traj_plot",
                    )

                    self.last_auc_mean.append(eval_dict[f"eval/mauc_mean"])
                    self.last_auc_std.append(eval_dict[f"eval/mauc_std"])

                    self.save_model(logging_step)

            torch.cuda.empty_cache()

        self.logger.print(
            "total dynamics model training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def evaluate(self):
        """
        Given one ref, show tracking performance
        """
        dimension = self.eval_env.pos_dimension
        assert dimension in [2, 3], "Dimension must be 2 or 3"

        # Set subplot parameters based on dimension
        if dimension == 3:
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2)  # 2D subplot
        else:
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
        tref_trajs = []
        ep_buffer = []
        for num_episodes in range(self.eval_episodes):
            ep_reward, ep_tracking_error, ep_control_effort, ep_inference_time = (
                0,
                0,
                0,
                0,
            )

            # Env initialization
            options = {"replace_x_0": True}
            obs, infos = self.eval_env.reset(seed=self.seed, options=options)

            tref_trajectory = [self.eval_env.time_steps]
            trajectory = [infos["x"][:dimension]]
            normalized_error_trajectory = [1.0]
            for t in range(1, self.eval_env.episode_len + 1):
                with torch.no_grad():
                    t0 = time.time()
                    a, _ = self.policy(obs, deterministic=True)
                    t1 = time.time()
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                next_obs, rew, term, trunc, infos = self.eval_env.step(a)

                tref_trajectory.append(self.eval_env.time_steps)
                trajectory.append(infos["x"][:dimension])  # Store trajectory point
                normalized_error_trajectory.append(infos["relative_tracking_error"])

                done = term or trunc

                obs = next_obs
                ep_reward += rew
                ep_inference_time += t1 - t0
                ep_tracking_error += infos["tracking_error"]
                ep_control_effort += infos["control_effort"]

                if done:
                    auc = np.trapezoid(normalized_error_trajectory, dx=self.eval_env.dt)
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
                    error_trajs.append(normalized_error_trajectory)
                    tref_trajs.append(tref_trajectory)

                    break

        # Optional: Add axis labels
        ax1.set_xlabel("X", labelpad=10)
        ax1.set_ylabel("Y", labelpad=10)
        if dimension == 3:
            ax1.set_zlabel("Z", labelpad=10)
            # Set a nice viewing angle for 3D
            ax1.view_init(elev=25, azim=45)

        # calculate the mean and std of the traj norm error to make plot
        i = 0
        for t, traj in zip(tref_trajs, error_trajs):
            ax2.plot(
                t,
                traj,
                alpha=0.7,
                c=COLORS[str(i)],
            )
            i += 1
        ax2.set_xlabel("Time Steps", labelpad=10)
        ax2.set_ylabel(r"$||x(t)-x^*(t)||_2 / ||x(0) - x^*(0)||_2$", labelpad=10)

        plt.tight_layout()
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())

        # Close the figure to free memory
        plt.close(fig)

        rew_list = [ep_info["avg_reward"] for ep_info in ep_buffer]
        inf_list = [ep_info["avg_inference_time"] for ep_info in ep_buffer]
        auc_list = [ep_info["mauc"] for ep_info in ep_buffer]
        trk_list = [ep_info["tracking_error"] for ep_info in ep_buffer]
        ctr_list = [ep_info["control_effort"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(rew_list), np.std(rew_list)
        inf_mean, inf_std = np.mean(inf_list), np.std(inf_list)
        auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)
        trk_mean, trk_std = np.mean(trk_list), np.std(trk_list)
        ctr_mean, ctr_std = np.mean(ctr_list), np.std(ctr_list)

        eval_dict = {
            f"eval/rew_mean": rew_mean,
            f"eval/rew_std": rew_std,
            f"eval/inf_mean": inf_mean,
            f"eval/inf_std": inf_std,
            f"eval/mauc_mean": auc_mean,
            f"eval/mauc_std": auc_std,
            f"eval/tracking_error_mean": trk_mean,
            f"eval/tracking_error_std": trk_std,
            f"eval/control_effort_mean": ctr_mean,
            f"eval/control_effort_std": ctr_std,
        }

        return eval_dict, image_array

    def save_model(self, e):
        ### save checkpoint
        name = f"model_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)

        model = (
            getattr(self.policy, "u_func", None)
            or getattr(self.policy, "actor", None)
            or self.policy
        )

        if model is not None:
            model = deepcopy(model).to("cpu")
            torch.save(model.state_dict(), path)

            # save the best model
            if (
                np.mean(self.last_auc_mean) < self.last_min_auc_mean
                and np.mean(self.last_auc_std) <= self.last_min_auc_std
            ):
                name = f"best_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(model.state_dict(), path)

                self.last_min_auc_mean = np.mean(self.last_auc_mean)
                self.last_min_auc_std = np.mean(self.last_auc_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str, name: str):
        image_list = [image]
        path_image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=path_image_path)

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


class DynamicsTrainer:
    def __init__(
        self,
        env: gym.Env,
        Dynamic_func: nn.Module,
        logger: WandbLogger,
        writer: SummaryWriter,
        buffer_size: int = 200_000,
        epochs: int = 10000,
    ) -> None:
        self.env = env
        self.Dynamic_func = Dynamic_func
        self.logger = logger
        self.writer = writer

        # training parameters
        self.buffer_size = buffer_size
        self.epochs = epochs

    def train(self) -> dict[str, float]:
        start_time = time.time()

        # Train loop
        data = self.env.get_rollout(self.buffer_size)
        self.Dynamic_func.train()
        with tqdm(
            total=self.epochs, desc=f"{self.Dynamic_func.name} Training (Epochs)"
        ) as pbar:
            while pbar.n < self.epochs:
                step = pbar.n + 1  # + 1 to avoid zero division

                # first sample batch (size of 1024) from the data
                batch = dict()
                indices = np.random.choice(self.buffer_size, size=128, replace=False)
                for key in data.keys():
                    # Sample a batch of 128
                    batch[key] = data[key][indices]

                loss_dict, update_time = self.Dynamic_func.learn(batch)

                # Calculate expected remaining time
                pbar.update(1)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.Dynamic_func.name}/analytics/timesteps"] = step
                loss_dict[f"{self.Dynamic_func.name}/analytics/update_time"] = (
                    update_time
                )

                self.write_log(loss_dict, step=step)

            torch.cuda.empty_cache()

        self.logger.print(
            "total dynamics model training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str, name: str):
        image_list = [image]
        path_image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=path_image_path)
