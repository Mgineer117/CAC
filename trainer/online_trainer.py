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
from trainer.base_trainer import BaseTrainer
from utils.sampler import OnlineSampler


# model-free policy trainer
class OnlineTrainer(BaseTrainer):
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

        self.seed = seed

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_auc_mean = deque(maxlen=3)

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

                loss_dict, supp_dict, update_time = self.policy.learn(batch)

                # Calculate expected remaining time
                pbar.update(batch["rewards"].shape[0])

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
                self.write_image(
                    supp_dict,
                    step=step,
                )

                #### EVALUATIONS ####
                if (step >= self.eval_interval * eval_idx) or (
                    pbar.n >= self.timesteps
                ):
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict_list = []
                    for i in range(self.eval_num):
                        eval_dict, supp_dict = self.evaluate()
                        eval_dict_list.append(eval_dict)

                    eval_dict = self.average_dict_values(eval_dict_list)

                    # Manual logging
                    self.write_log(eval_dict, step=step, eval_log=True)
                    self.write_image(supp_dict, step=step)

                    self.last_auc_mean.append(eval_dict[f"eval/mauc_mean"])

                    self.save_model(step)

            torch.cuda.empty_cache()

        self.logger.print(
            "total PPO training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

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
            if np.mean(self.last_auc_mean) < self.last_min_auc_mean:
                name = f"best_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(model.state_dict(), path)

                self.last_min_auc_mean = np.mean(self.last_auc_mean)
        else:
            raise ValueError("Error: Model is not identifiable!!!")
