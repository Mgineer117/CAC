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
class Trainer:
    def __init__(
        self,
        env: gym.Env,
        policy: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        timesteps: int = 1e6,
        log_interval: int = 2,
        eval_num: int = 10,
        eval_episodes: int = 10,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.eval_num = eval_num
        self.eval_episodes = eval_episodes

        self.logger = logger
        self.writer = writer

        # training parameters
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

        self.last_auc_mean = deque(maxlen=1)
        self.last_auc_std = deque(maxlen=1)

        # Train loop
        eval_idx = 0
        with tqdm(
            total=self.timesteps, desc=f"{self.policy.name} Training (Timesteps)"
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
                    self.last_auc_std.append(eval_dict[f"eval/mauc_std"])

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
        dimension = self.env.pos_dimension
        assert dimension in [2, 3], "Dimension must be 2 or 3"

        # Set subplot parameters based on dimension
        if dimension == 3:
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2)  # 2D subplot
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

        # Dynamically create the coordinate list and plot the reference trajectory
        coords = [self.env.xref[:, i] for i in range(dimension)]
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
            obs, infos = self.env.reset(seed=self.seed, options=options)

            tref_trajectory = [self.env.time_steps]
            trajectory = [infos["x"][:dimension]]
            normalized_error_trajectory = [1.0]
            for t in range(1, self.env.episode_len + 1):
                with torch.no_grad():
                    t0 = time.time()
                    a, _ = self.policy(obs, deterministic=True)
                    t1 = time.time()
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                next_obs, rew, term, trunc, infos = self.env.step(a)

                tref_trajectory.append(self.env.time_steps)
                trajectory.append(infos["x"][:dimension])  # Store trajectory point
                normalized_error_trajectory.append(infos["relative_tracking_error"])

                done = term or trunc

                obs = next_obs
                ep_reward += rew
                ep_inference_time += t1 - t0
                ep_tracking_error += infos["tracking_error"]
                ep_control_effort += infos["control_effort"]

                if done:
                    auc = np.trapezoid(normalized_error_trajectory, dx=self.env.dt)
                    ep_buffer.append(
                        {
                            "avg_reward": ep_reward / t,
                            "avg_inference_time": ep_inference_time / t,
                            "mauc": auc * (self.env.episode_len / t),
                            "tracking_error": ep_tracking_error / t,
                            "control_effort": ep_control_effort / t,
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

        convergence_rate, overshoot = self.compute_convergence_rate(
            tref_trajs, error_trajs
        )

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
            f"eval/convergence_rate": convergence_rate,
            f"eval/overshoot": overshoot,
        }

        return eval_dict, image_array

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

    def compute_convergence_rate(self, tref_trajs, error_trajs, alpha=0.05):
        # -----------------------------------------------------------
        # Step A: Find (lambda*, C*) that minimize the AUC on the FIRST trajectory
        # -----------------------------------------------------------
        # Convert first trajectory time to a NumPy array (if it's not already)
        # Step 1: Find the trajectory with the highest overshoot
        overshoots = [np.max(np.abs(e)) for e in error_trajs]
        idx_max = np.argmax(overshoots)

        t_ref = np.array(tref_trajs[idx_max], dtype=np.float64)
        e_ref = np.abs(error_trajs[idx_max])  # Take absolute to match upper-bound logic
        T = np.max(t_ref)

        # Step 2: Define the area under exponential bound
        def area_of_lambda(lmbd):
            temp = np.clip(lmbd * t_ref, None, 700)
            c_val = max(1.0, np.max(e_ref * np.exp(temp)))
            return c_val * (1.0 - np.exp(-lmbd * T)) / lmbd

        lambdas = np.logspace(-4, 2, 200)
        areas = [area_of_lambda(l) for l in lambdas]
        idx_min = np.argmin(areas)
        lambda_star = lambdas[idx_min]
        C_star = max(1.0, np.max(e_ref * np.exp(lambda_star * t_ref)))

        # -----------------------------------------------------------
        # Step B: For each trajectory, find the largest lambda_i s.t. e(t) <= C_star e^{-lambda_i t}
        # -----------------------------------------------------------
        def feasible_lambda_for_trajectory(t, e, c_val):
            """
            Solve for the largest lambda_i satisfying
                e[j] <= c_val * exp(-lambda_i * t[j]) for all j.
            =>  e[j] * exp(lambda_i * t[j]) <= c_val
            =>  lambda_i <= (1/t[j]) * ln(c_val / e[j])  for e[j]>0, t[j]>0
            """
            mask = (e > 0) & (t > 0)
            if not np.any(mask):
                # If e is zero for all t or t=0, no constraint => lambda_i can be "infinite"
                return np.inf

            e_nonzero = e[mask]
            t_nonzero = t[mask]
            bounds = (
                np.log(c_val / e_nonzero) / t_nonzero
            )  # might be inf if e_nonzero < c_val
            return np.min(bounds)

        lambda_ratios = []
        # Convert each t_i to float array if needed, e_i is already a NumPy array
        for t_i, e_i in zip(tref_trajs, error_trajs):
            t_i = np.array(t_i, dtype=float)
            e_i = np.array(e_i, dtype=float)
            lam_i = feasible_lambda_for_trajectory(t_i, e_i, C_star)
            # ratio_i = lam_i / C_star
            lambda_ratios.append(lam_i)

        # -----------------------------------------------------------
        # Step C: Compute the (1 - alpha)-quantile of these ratios
        # -----------------------------------------------------------
        # e.g., alpha=0.05 => 95th percentile
        lambda_quantile = np.quantile(lambda_ratios, 1 - alpha)

        return lambda_quantile, C_star
