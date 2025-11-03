import time
from datetime import date
from math import ceil, floor
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from utils.misc import temp_seed

today = date.today()


class Base:
    def __init__():
        pass

    def get_reset_data(self, batch_size):
        """
        We create a initialization batch to avoid the daedlocking.
        The remainder of zero arrays will be cut in the end.
        np.nan makes it easy to debug
        """
        data = dict(
            states=np.full(((batch_size, self.state_dim)), np.nan, dtype=np.float32),
            next_states=np.full(
                ((batch_size, self.state_dim)), np.nan, dtype=np.float32
            ),
            actions=np.full((batch_size, self.action_dim), np.nan, dtype=np.float32),
            rewards=np.full((batch_size, 1), np.nan, dtype=np.float32),
            terminals=np.full((batch_size, 1), np.nan, dtype=np.float32),
            logprobs=np.full((batch_size, 1), np.nan, dtype=np.float32),
            entropys=np.full((batch_size, 1), np.nan, dtype=np.float32),
        )
        return data


class OnlineSampler(Base):
    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        episode_len: int,
        batch_size: int,
    ) -> None:
        super(Base, self).__init__()
        """
        This computes the ""very"" appropriate parameter for the Monte-Carlo sampling
        given the number of episodes and the given number of cores the runner specified.
        ---------------------------------------------------------------------------------
        Rounds: This gives several rounds when the given sampling load exceeds the number of threads
        the task is assigned. 
        This assigned appropriate parameters assuming one worker work with 2 trajectories.
        """

        # dimensional params
        self.state_dim = state_dim
        self.action_dim = action_dim

        # sampling params
        self.episode_len = episode_len
        self.batch_size = batch_size

        self.episodes_per_worker = 3  # 3 episodes per worker for efficiency
        self.thread_batch_size = self.episodes_per_worker * self.episode_len
        self.total_num_worker = ceil(self.batch_size / (self.thread_batch_size))

        self.manager = mp.Manager()
        self.queue = self.manager.Queue()

        # enforce one thread for each worker to avoid CPU overscription.
        torch.set_num_threads(1)

    def collect_samples(
        self,
        env,
        policy,
        seed: int,
        deterministic: bool = False,
    ):
        """
        Collect samples in parallel using multiprocessing.

        Args:
            env: The environment to interact with.
            policy: Policy to sample actions from.
            seed (int | None): Seed for reproducibility.
            deterministic (bool): Whether to use deterministic policy.
            random_init_pos (bool): Randomize initial position in env reset.

        Returns:
            memory (dict): Sampled batch.
            duration (float): Time taken to collect.
        """
        t_start = time.time()
        device = next((p.device for p in policy.parameters()), torch.device("cpu"))

        policy.to_device(torch.device("cpu"))

        processes = []
        worker_memories = [None] * self.total_num_worker
        for i in range(self.total_num_worker):
            args = (
                i,
                self.queue,
                env,
                policy,
                seed,
                deterministic,
            )
            p = mp.Process(target=self.collect_trajectory, args=args)
            processes.append(p)
            p.start()

        # ✅ Wait for just the subprocess workers of this round
        expected = len(processes)
        collected = 0
        while collected < expected:
            try:
                pid, data = self.queue.get(timeout=300)
                if worker_memories[pid] is None:
                    worker_memories[pid] = data
                    collected += 1
            except Empty:
                print(f"[Warning] Queue timeout. Retrying... ({collected}/{expected})")

        start_time = time.time()
        for p in processes:
            p.join(timeout=max(0.1, 10 - (time.time() - start_time)))
            if p.is_alive():
                p.terminate()
                p.join()  # Force cleanup

        # ✅ Merge memory
        memory = {}
        for wm in worker_memories:
            if wm is None:
                raise RuntimeError("One or more workers failed to return data.")
            for key, val in wm.items():
                if key in memory:
                    memory[key] = np.concatenate((memory[key], wm[key]), axis=0)
                else:
                    memory[key] = wm[key]

        # # ✅ Truncate to desired batch size
        # for k in memory:
        #     memory[k] = memory[k][: self.batch_size]

        t_end = time.time()
        policy.to_device(device)

        return memory, t_end - t_start

    def collect_trajectory(
        self,
        pid,
        queue,
        env,
        policy: nn.Module,
        seed: int,
        deterministic: bool = False,
    ):
        # estimate the batch size to hava a large batch
        data = self.get_reset_data(
            batch_size=self.thread_batch_size + self.episode_len
        )  # allocate memory
        seed = temp_seed(seed, pid)

        current_step = 0
        for i in range(self.episodes_per_worker):
            # env initialization
            obs, _ = env.reset(seed=seed + i)

            for t in range(self.episode_len):
                with torch.no_grad():
                    a, metaData = policy(obs, deterministic=deterministic)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                    # env stepping
                    next_obs, rew, term, trunc, infos = env.step(a)
                    done = term or trunc

                # saving the data
                data["states"][current_step + t] = obs
                data["next_states"][current_step + t] = next_obs
                data["actions"][current_step + t] = a
                data["rewards"][current_step + t] = rew
                data["terminals"][current_step + t] = done
                data["logprobs"][current_step + t] = (
                    metaData["logprobs"].detach().numpy()
                )
                data["entropys"][current_step + t] = (
                    metaData["entropy"].detach().numpy()
                )

                if done:
                    # clear log
                    current_step += t + 1
                    break

                obs = next_obs

        for k in data:
            data[k] = data[k][:current_step]

        return queue.put([pid, data]) if queue is not None else data
