import os
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.base import Base
from trainer.base_trainer import BaseTrainer


class C3MTrainer(BaseTrainer):
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
        super().__init__(
            env=env,
            eval_env=eval_env,
            policy=policy,
            logger=logger,
            writer=writer,
            init_epochs=init_epochs,
            epochs=epochs,
            log_interval=log_interval,
            eval_num=eval_num,
            eval_episodes=eval_episodes,
            seed=seed,
        )

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_auc_mean = deque(maxlen=3)

        # Train loop
        batch_size = 2048
        eval_idx = 0
        self.policy.train()
        with tqdm(
            initial=self.init_epochs,
            total=(self.epochs + self.init_epochs),
            desc=f"{self.policy.name} Training (Epochs)",
        ) as pbar:
            while pbar.n < (self.epochs + self.init_epochs):
                step = pbar.n + 1  # + 1 to avoid zero division
                logging_step = (step - self.init_epochs) * batch_size + self.init_epochs

                loss_dict, supp_dict, update_time = self.policy.learn()

                # Calculate expected remaining time
                pbar.update(1)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.policy.name}/analytics/timesteps"] = logging_step
                loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time

                self.write_log(loss_dict, step=logging_step)
                self.write_image(
                    supp_dict,
                    step=logging_step,
                )

                #### EVALUATIONS ####
                if step >= (self.eval_interval * eval_idx + self.init_epochs):
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict_list = []
                    for i in range(self.eval_num):
                        eval_dict, supp_dict = self.evaluate()
                        eval_dict_list.append(eval_dict)

                    eval_dict = self.average_dict_values(eval_dict_list)

                    # Manual logging
                    self.write_log(eval_dict, step=logging_step, eval_log=True)
                    self.write_image(supp_dict, step=logging_step)

                    self.last_auc_mean.append(eval_dict[f"eval/mauc_mean"])

                    self.save_model(logging_step)

            torch.cuda.empty_cache()

        self.logger.print(
            "total dynamics model training time: {:.2f} hours".format(
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


class DynamicsTrainer(BaseTrainer):
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
        data = self.env.get_rollout(self.buffer_size, mode="dynamics")
        self.Dynamic_func.train()
        with tqdm(
            total=self.epochs, desc=f"{self.Dynamic_func.name} Training (Epochs)"
        ) as pbar:
            while pbar.n < self.epochs:
                step = pbar.n + 1  # + 1 to avoid zero division

                # first sample batch (size of 1024) from the data
                batch = dict()
                indices = np.random.choice(self.buffer_size, size=1024, replace=False)
                for key in data.keys():
                    # Sample a batch of 1024
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


class SDCTrainer(BaseTrainer):
    def __init__(
        self,
        env: gym.Env,
        SDC_func: nn.Module,
        logger: WandbLogger,
        writer: SummaryWriter,
        buffer_size: int = 10_000,
        init_epochs: int = 0,
        epochs: int = 10000,
    ) -> None:
        self.env = env
        self.SDC_func = SDC_func
        self.logger = logger
        self.writer = writer

        # training parameters
        self.buffer_size = buffer_size
        self.init_epochs = init_epochs
        self.epochs = epochs

    def train(self) -> dict[str, float]:
        start_time = time.time()

        # Train loop
        batch_size = 1024
        data = self.env.get_rollout(self.buffer_size, mode="c3m")
        self.SDC_func.train()
        with tqdm(
            initial=self.init_epochs,
            total=(self.epochs + self.init_epochs),
            desc=f"{self.SDC_func.name} Training (Epochs)",
        ) as pbar:
            while pbar.n < (self.epochs + self.init_epochs):
                step = pbar.n + 1  # + 1 to avoid zero division

                # first sample batch (size of 1024) from the data
                batch = dict()
                indices = np.random.choice(
                    self.buffer_size, size=batch_size, replace=False
                )
                for key in data.keys():
                    # Sample a batch of 1024
                    batch[key] = data[key][indices]

                loss_dict, update_time = self.SDC_func.learn(batch)

                # Calculate expected remaining time
                pbar.update(1)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.SDC_func.name}/analytics/update_time"] = update_time

                self.write_log(loss_dict, step=step)

            torch.cuda.empty_cache()

        self.logger.print(
            "total sdc model training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )
