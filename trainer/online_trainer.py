import os
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.base import Base
from utils.sampler import OnlineSampler

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


def check_batch(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN detected in batch[{key}]")
                return True
            if torch.isinf(value).any():
                print(f"Inf detected in batch[{key}]")
                return True
    return False


def check_network(network):
    for name, param in network.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in {name}")
            return True
        if torch.isinf(param).any():
            print(f"Inf detected in {name}")
            return True
    return False


# model-free policy trainer
class OnlineTrainer:
    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        policy: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_epochs: int = 0,
        timesteps: int = 1e6,
        log_interval: int = 2,
        eval_num: int = 10,
        eval_episodes: int = 10,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        self.sampler = sampler
        self.eval_num = eval_num
        self.eval_episodes = eval_episodes

        self.logger = logger
        self.writer = writer

        # training parameters
        self.init_epochs = init_epochs
        self.timesteps = timesteps
        self.nupdates = self.policy.nupdates

        self.log_interval = log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)

        # initialize the essential training components
        self.last_min_auc_mean = 1e10
        self.last_min_auc_std = 1e10

        self.seed = seed

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_auc_mean = deque(maxlen=3)
        self.last_auc_std = deque(maxlen=3)

        # Train loop
        eval_idx = 0
        with tqdm(
            initial=self.init_epochs,
            total=self.timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < self.timesteps:
                step = pbar.n + 1  # + 1 to avoid zero division

                self.policy.train()
                batch, sample_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=self.seed
                )

                loss_dict, ppo_timesteps, update_time = self.policy.learn(batch)

                # Calculate expected remaining time
                pbar.update(ppo_timesteps)

                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / step
                remaining_time = avg_time_per_iter * (self.timesteps - step)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.policy.name}/analytics/timesteps"] = step
                loss_dict[f"{self.policy.name}/analytics/sample_time"] = sample_time
                loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time
                loss_dict[f"{self.policy.name}/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss_dict, step=step)

                #### EVALUATIONS ####
                if (step >= self.eval_interval * eval_idx) or (
                    pbar.n >= self.timesteps
                ):
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict_list = []
                    for i in range(self.eval_num):
                        eval_dict, traj_plot = self.evaluate()
                        eval_dict_list.append(eval_dict)

                    eval_dict = self.average_dict_values(eval_dict_list)

                    # Manual logging
                    self.write_log(eval_dict, step=step, eval_log=True)
                    self.write_image(
                        traj_plot,
                        step=step,
                        logdir=f"eval",
                        name="traj_plot",
                    )

                    self.last_auc_mean.append(eval_dict[f"eval/mauc_mean"])
                    self.last_auc_std.append(eval_dict[f"eval/mauc_std_(95)"])

                    self.save_model(step)

            torch.cuda.empty_cache()

        self.logger.print(
            "total PPO training time: {:.2f} hours".format(
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
        }

        return eval_dict, image_array

    def mean_confidence_interval(self, data, confidence=0.95):
        n = len(data)
        data = np.array(data)
        mean = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(n)  # standard error
        h = 1.96 * sem  # margin of error for 95% CI
        return mean, h

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
