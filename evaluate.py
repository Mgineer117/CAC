import glob
import math
import os
import time

import numpy as np
import torch

from envs.car import CarEnv
from envs.neurallander import NeuralLanderEnv
from envs.pvtol import PvtolEnv
from envs.quadrotor import QuadRotorEnv
from envs.turtlebot import TurtlebotEnv
from policy.layers.c3m_networks import C3M_U
from policy.layers.dynamic_networks import DynamicLearner
from policy.layers.ppo_networks import PPO_Actor
from policy.layers.sd_lqr_networks import SDCLearner
from policy.lqr import LQR, LQR_Approximation
from policy.sd_lqr import SD_LQR
from utils.misc import seed_all


def trim_state(state: torch.Tensor):
    state = torch.from_numpy(state).cpu()
    if len(state.shape) == 1:
        state = state.unsqueeze(0)

    # state trimming
    x = state[:, :x_dim]  # .requires_grad_()
    xref = state[:, x_dim:-action_dim]  # .requires_grad_()
    uref = state[:, -action_dim:]  # .requires_grad_()

    x_trim = x[:, effective_indices]  # .requires_grad_()
    xref_trim = xref[:, effective_indices]  # .requires_grad_()

    return x, xref, uref, x_trim, xref_trim


def average_and_sem_dict(dict_list):
    if not dict_list:
        return {}, {}

    keys = dict_list[0].keys()
    n = len(dict_list)

    # Step 1: Compute mean
    sum_dict = {key: 0.0 for key in keys}
    for d in dict_list:
        for key, value in d.items():
            sum_dict[key] += value
    mean_dict = {key: total / n for key, total in sum_dict.items()}

    # Step 2: Compute 95% confidence interval
    squared_diff = {key: 0.0 for key in keys}
    for d in dict_list:
        for key in keys:
            squared_diff[key] += (d[key] - mean_dict[key]) ** 2

    sem_dict = {
        key: 1.96 * math.sqrt(squared_diff[key] / n) / math.sqrt(n) for key in keys
    }

    return mean_dict, sem_dict


def get_env(env_name, sigma: float = 0.0):
    if env_name == "car":
        env = CarEnv(sigma=sigma)
    elif env_name == "pvtol":
        env = PvtolEnv(sigma=sigma)
    elif env_name == "quadrotor":
        env = QuadRotorEnv(sigma=sigma)
    elif env_name == "neurallander":
        env = NeuralLanderEnv(sigma=sigma)
    elif env_name == "turtlebot":
        env = TurtlebotEnv(sigma=sigma)
    else:
        raise NotImplementedError(f"{env_name} is not implemented.")

    global x_dim, state_dim, action_dim, effective_indices, episode_len

    x_dim = env.num_dim_x
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    effective_indices = env.effective_indices
    episode_len = env.episode_len

    return env


def get_policy(env_name, algo_name: str, model_dir: str):
    if algo_name == "lqr":
        policy = LQR(
            x_dim=x_dim,
            effective_indices=effective_indices,
            action_dim=action_dim,
            f_func=env.f_func,
            B_func=env.B_func,
            Bbot_func=env.Bbot_func,
        )
    else:
        if algo_name in ("c3m", "c3m_approx"):
            policy = C3M_U(
                x_dim=x_dim,
                state_dim=state_dim,
                effective_indices=effective_indices,
                action_dim=action_dim,
                task=env_name,
            )
        elif algo_name in ("ppo", "cac", "cac_approx", "cac_approx_deterministic"):
            policy = PPO_Actor(
                input_dim=state_dim, hidden_dim=[64, 64], a_dim=action_dim
            )
        elif algo_name == "lqr_approx":
            Dynamic_func = DynamicLearner(
                x_dim=x_dim,
                action_dim=action_dim,
                hidden_dim=[256, 256],
                drop_out=0.2,
            )
            policy = LQR_Approximation(
                x_dim=x_dim,
                effective_indices=effective_indices,
                action_dim=action_dim,
                Dynamic_func=Dynamic_func,
            )
        elif algo_name == "sd-lqr":
            Dynamic_func = DynamicLearner(
                x_dim=x_dim,
                action_dim=action_dim,
                hidden_dim=[256, 256],
                drop_out=0.2,
            )
            SDC_func = SDCLearner(
                x_dim=x_dim,
                a_dim=action_dim,
                hidden_dim=[256, 256],
            )
            policy = SD_LQR(
                x_dim=x_dim,
                effective_indices=effective_indices,
                action_dim=action_dim,
                Dynamic_func=Dynamic_func,
                SDC_func=SDC_func,
            )

        policy.load_state_dict(torch.load(model_dir, map_location=torch.device("cpu")))
    return policy.to(torch.float32).cpu()


def evaluate(env, policy, eval_episodes):
    ep_buffer = []
    for num_episodes in range(eval_episodes):
        ep_reward, ep_tracking_error, ep_control_effort, ep_inf = 0, 0, 0, 0

        # Env initialization
        if num_episodes == 0:
            options = None
        else:
            options = {"replace_x_0": True}
        obs, infos = env.reset(seed=seed, options=options)

        normalized_error_trajectory = [1.0]
        for t in range(1, env.episode_len + 1):
            with torch.no_grad():
                try:
                    x, xref, uref, x_trim, xref_trim = trim_state(obs)
                    t0 = time.time()
                    a, _ = policy(x, xref, uref, x_trim, xref_trim, deterministic=True)
                    inf_time = time.time() - t0
                except:
                    t0 = time.time()
                    a, _ = policy(obs, deterministic=True)
                    inf_time = time.time() - t0
                a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

            next_obs, rew, term, trunc, infos = env.step(a)
            done = term or trunc

            normalized_error_trajectory.append(infos["relative_tracking_error"])

            obs = next_obs
            ep_reward += rew
            ep_tracking_error += infos["tracking_error"]
            ep_control_effort += infos["control_effort"]
            ep_inf += inf_time

            if done:
                mauc = np.trapezoid(normalized_error_trajectory, dx=env.dt)
                ep_buffer.append(
                    {
                        "avg_reward": ep_reward / t,
                        "mauc": (mauc / t) * env.episode_len,
                        "tracking_error": ep_tracking_error / t,
                        "control_effort": ep_control_effort / t,
                        "inference_time": ep_inf / t,
                    }
                )

                break

    rew_list = [ep_info["avg_reward"] for ep_info in ep_buffer]
    mauc_list = [ep_info["mauc"] for ep_info in ep_buffer]
    trk_list = [ep_info["tracking_error"] for ep_info in ep_buffer]
    ctr_list = [ep_info["control_effort"] for ep_info in ep_buffer]
    inf_list = [ep_info["inference_time"] for ep_info in ep_buffer]

    rew_mean, rew_std = np.mean(rew_list), np.std(rew_list)
    mauc_mean, mauc_std = np.mean(mauc_list), np.std(mauc_list)
    trk_mean, trk_std = np.mean(trk_list), np.std(trk_list)
    ctr_mean, ctr_std = np.mean(ctr_list), np.std(ctr_list)
    inf_mean, inf_std = np.mean(inf_list), np.std(inf_list)

    eval_dict = {
        f"rew_mean": rew_mean,
        f"rew_std": rew_std,
        f"mauc_mean": mauc_mean,
        f"mauc_std": mauc_std,
        f"trk_error_mean": trk_mean,
        f"trk_error_std": trk_std,
        f"ctr_effort_mean": ctr_mean,
        f"ctr_effort_std": ctr_std,
        f"inf_mean": inf_mean,
        f"inf_std": inf_std,
    }
    return eval_dict


def run(env, policy, seed):
    # fix seed
    seed_all(seed)

    eval_num = 5
    eval_episodes = 5
    eval_dict_list = []
    for _ in range(eval_num):
        eval_dict = evaluate(env, policy, eval_episodes)
        eval_dict_list.append(eval_dict)

    eval_dict, _ = average_and_sem_dict(eval_dict_list)
    return eval_dict


if __name__ == "__main__":
    """
    Instruction:
    This script loads and evaluates trained models in the "model/" directory for the specified
    environment ("car") and algorithm ("cac"). It computes the mean and standard error of a
    performance metric (e.g., MAUC) across different seeds and prints the summary.

    Expected model filename format: <prefix>_<seed>.pth
    """

    torch.set_default_dtype(torch.float32)

    env_name = "car"
    algo_name = "cac"

    base_dir = f"model/"

    env = get_env(env_name)
    model_dirs = sorted(glob.glob(os.path.join(base_dir, "*.pth")))
    num_models = len(model_dirs)

    print(f"-------------------------------------------------------")
    print(f"      ENV NAME      : {env_name}")
    print(f"      ALSO NAME     : {algo_name}")
    print(f"      NUM. MODELS   : {num_models}")
    print(f"-------------------------------------------------------")

    eval_dict_list = []
    for i, model_dir in enumerate(model_dirs):
        policy = get_policy(env_name, algo_name, model_dir)
        seed = int(model_dir.split("_")[-1][:-4])
        eval_dict = run(env, policy, seed)
        eval_dict_list.append(eval_dict)
        print(
            f"====== Running {i} | Seed {seed}: mauc: {eval_dict["mauc_mean"]:3f}====="
        )

    eval_dict, semd_dict = average_and_sem_dict(eval_dict_list)

    print(f"MEAN: {eval_dict}")
    print(f"STD ERROR: {semd_dict}")
