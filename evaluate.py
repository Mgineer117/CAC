# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
import datetime
import json
import os
import random
import time
import uuid

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

import wandb
from policy.layers.dynamic_networks import DynamicLearner
from policy.layers.sd_lqr_networks import SDCLearner
from trainer.evaluator import Evaluator
from utils.get_args import get_args
from utils.get_dynamics import get_dynamics
from utils.get_sdc import get_SDC
from utils.misc import (
    concat_csv_columnwise_and_delete,
    override_args,
    seed_all,
    setup_logger,
)
from utils.utils import call_env

os.environ["WANDB_MODE"] = "disabled"

from policy.layers.c3m_networks import C3M_U, C3M_W, C3M_U_Gaussian
from policy.lqr import LQR
from policy.sd_lqr import SD_LQR

EVAL_EPISODES = 10
COLORS = {
    "cac-approx": "#F83C32",  # pastel red
    "cac-approx-deterministic": "#F88A84",  # pale pastel red
    "cac-approx-noentropy": "#F88A84",  # pale pastel red
    "c3m-approx": "#1A62CF",  # pastel blue
    "ppo-approx": "#646464",  # grey
    "lqr-approx": "#138B08",  # green
    "sd-lqr-approx": "#C00FE4",  # purple
}

LINESTYLES = {
    "cac-approx": "-",
    "cac-approx-deterministic": "-",
    "cac-approx-noentropy": "-",
    "c3m-approx": "-.",
    "ppo-approx": ":",
    "lqr-approx": "--",
    "sd-lqr-approx": "--",
}
# NAMES = {
#     "cac-approx": r"CAC w/ $\mathcal{H}(\pi(\cdot\mid s))$",
#     "cac-approx-deterministic": r"CAC w/o $\mathcal{H}(\pi(\cdot\mid s))$",
#     "c3m-approx": "C3M",
#     "ppo-approx": "PPO",
# }

# LIMITS = {
#     "car": {"x"; [], "y": [-5, 5], "z": [-1, 1]},
#     "pvtol": 50.0,
#     "neurallander": 10.0,
#     "quadrotor": 15.0,
# }

NAMES_CAR = {
    "cac-approx": 1.07,
    "cac-approx-deterministic": 1.08,
    "cac-approx-noentropy": 1.08,
    "c3m-approx": 1.06,
    "ppo-approx": 1.56,
    "sd-lqr-approx": 7.88,
    "lqr-approx": 9.60,
}
NAMES_PVTOL = {
    "cac-approx": 7.49,
    "cac-approx-deterministic": 8.79,
    "cac-approx-noentropy": 7.91,
    "c3m-approx": 29.6,
    "ppo-approx": 12.7,
    "sd-lqr-approx": 16.7,
    "lqr-approx": 17.7,
}
NAMES_NEURALLANDER = {
    "cac-approx": 2.47,
    "cac-approx-deterministic": 2.51,
    "cac-approx-noentropy": 2.54,
    "c3m-approx": 2.63,
    "ppo-approx": 2.64,
    "sd-lqr-approx": 5.67,
    "lqr-approx": 5.70,
}
NAMES_QUADROTOR = {
    "cac-approx": 5.55,
    "cac-approx-deterministic": 5.64,
    "cac-approx-noentropy": 5.29,
    "c3m-approx": 6.97,
    "ppo-approx": 5.84,
    "sd-lqr-approx": 5.73,
    "lqr-approx": 5.79,
}
NAMES = {
    "car": NAMES_CAR,
    "pvtol": NAMES_PVTOL,
    "neurallander": NAMES_NEURALLANDER,
    "quadrotor": NAMES_QUADROTOR,
}
MAX_TIME = {"car": 6.0, "pvtol": 6.0, "neurallander": 3.0, "quadrotor": 6.0}


class Policy:
    def __init__(self, x_dim, action_dim, policy):
        self.x_dim = x_dim
        self.action_dim = action_dim
        self.policy = policy

    def __call__(self, obs, deterministic=False):
        obs = torch.from_numpy(obs).unsqueeze(0).to(torch.float32)
        x, xref, uref, _ = self.trim_state(obs)
        return self.policy(x, xref, uref, deterministic=deterministic)

    def trim_state(self, state: torch.Tensor):
        # state trimming
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : 2 * self.x_dim]
        uref = state[:, 2 * self.x_dim : 2 * self.x_dim + self.action_dim]
        t = state[:, -1]

        return x, xref, uref, t


def smooth(
    scalars: list[float], weight: float = 0.9
) -> list[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def get_policy(eval_env, args, get_f_and_B, SDC_func, seed):
    env_name = args.task

    model_dir = f"model/{env_name}/{args.algo_name}/{seed}.pth"

    if args.algo_name in ("lqr", "lqr-approx", "sd-lqr", "sd-lqr-approx"):

        if args.algo_name in ("lqr", "lqr-approx"):
            policy = LQR(
                x_dim=eval_env.num_dim_x,
                action_dim=args.action_dim,
                get_f_and_B=get_f_and_B,
            )
        elif args.algo_name in ("sd-lqr", "sd-lqr-approx"):
            policy = SD_LQR(
                x_dim=eval_env.num_dim_x,
                action_dim=args.action_dim,
                get_f_and_B=get_f_and_B,
                SDC_func=SDC_func,
            )

    elif args.algo_name in (
        "ppo",
        "ppo-approx",
        "cac",
        "cac-approx",
        "cac-approx-deterministic",
        "cac-approx-noentropy",
    ):
        policy = C3M_U_Gaussian(
            x_dim=eval_env.num_dim_x,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            task=args.task,
        )
        policy.load_state_dict(torch.load(model_dir))
        policy = Policy(eval_env.num_dim_x, args.action_dim, policy)

    elif args.algo_name in ("c3m", "c3m-approx"):
        policy = C3M_U(
            x_dim=eval_env.num_dim_x,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            task=args.task,
        )
        policy.load_state_dict(torch.load(model_dir))
        policy = Policy(eval_env.num_dim_x, args.action_dim, policy)

    return policy


def evaluate(eval_env, algo_name, policy, seed):
    """
    Given one ref, show tracking performance
    """

    ep_buffer, ep_norm_tracking_error = [], []
    x_list = []
    for i in range(EVAL_EPISODES):
        ep_tracking_error, ep_control_effort, ep_inference_time = (
            0,
            0,
            0,
        )

        # Env initialization
        options = {"replace_x_0": True}
        obs, infos = eval_env.reset(seed=seed, options=options)

        x = [obs[: eval_env.num_dim_x]]

        normalized_error_trajectory = [1.0]
        for t in range(1, eval_env.episode_len + 1):
            with torch.no_grad():
                t0 = time.time()
                a, _ = policy(obs, deterministic=True)
                t1 = time.time()
                a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

            next_obs, rew, term, trunc, infos = eval_env.step(a)
            done = term or trunc

            obs = next_obs

            ep_inference_time += t1 - t0
            normalized_error_trajectory.append(infos["relative_tracking_error"])
            ep_tracking_error += infos["tracking_error"]
            ep_control_effort += infos["control_effort"]
            x.append(obs[: eval_env.num_dim_x])

            if done:
                auc = np.trapezoid(normalized_error_trajectory, dx=eval_env.dt)
                ep_norm_tracking_error.append(
                    np.array(smooth(normalized_error_trajectory))
                )

                ep_buffer.append(
                    {
                        "avg_inference_time": ep_inference_time / t,
                        "mauc": auc * (eval_env.episode_len / t),
                        "tracking_error": ep_tracking_error / t,
                        "control_effort": ep_control_effort / t,
                        "episode_len": t + 1,
                    }
                )
                x_list.append(x)

                break

    inf_mean = np.mean([ep_info["avg_inference_time"] for ep_info in ep_buffer])
    mauc_mean = np.mean([ep_info["mauc"] for ep_info in ep_buffer])
    trk_mean = np.mean([ep_info["tracking_error"] for ep_info in ep_buffer])
    ctr_mean = np.mean([ep_info["control_effort"] for ep_info in ep_buffer])

    eval_dict = {
        f"{algo_name}/inf_mean": inf_mean,
        f"{algo_name}/mauc_mean": mauc_mean,
        f"{algo_name}/tracking_error_mean": trk_mean,
        f"{algo_name}/control_effort_mean": ctr_mean,
    }

    # find the average of normalized_tracking error
    # but ensure they are in different length so extend the shorter length to the maximum length by copying the last element
    max_length = max(len(traj) for traj in ep_norm_tracking_error)
    ep_norm_tracking_error = np.array(
        [
            np.pad(traj, (0, max_length - len(traj)), mode="edge")
            for traj in ep_norm_tracking_error
        ]
    )
    ep_norm_tracking_error = np.mean(ep_norm_tracking_error, axis=0)

    # save tracking figures to fig/task/tracking/
    if not os.path.exists(f"fig/{eval_env.task}/tracking/{algo_name}"):
        os.makedirs(f"fig/{eval_env.task}/tracking/{algo_name}")

    if eval_env.pos_dimension == 2:
        plt.figure()
        # indicate the start and end points
        for traj in x_list:
            traj = np.array(traj)
            plt.plot(traj[:, 0], traj[:, 1], alpha=0.8, linewidth=3)
            plt.scatter(traj[0, 0], traj[0, 1], s=100, marker="*", alpha=0.8)

        plt.plot(
            eval_env.xref[:, 0],
            eval_env.xref[:, 1],
            color="black",
            linewidth=2,
            linestyle="--",
            alpha=0.6,
        )
        plt.scatter(
            eval_env.xref[0, 0],
            eval_env.xref[0, 1],
            color="black",
            s=100,
            marker="*",
            label="Start",
            alpha=0.6,
        )

        plt.xlabel("X position", fontsize=16)
        plt.ylabel("Z position", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.axis("equal")
        # plt.xlim([-7, 2.5])
        # plt.ylim([2.0, 9.0])
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(
            f"fig/{eval_env.task}/tracking/{algo_name}/{algo_name}({seed})_tracking_trajectories.svg"
        )
        plt.close()
    elif eval_env.pos_dimension == 3:
        plt.figure()
        ax = plt.axes(projection="3d")

        # Plot simulated trajectories
        for traj in x_list:
            traj = np.array(traj)
            ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.8, linewidth=3)
            # Add start/end dots for each trajectory
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], alpha=0.8, s=100, marker="*")

        # Plot reference trajectory (consistent with 2D style)
        ax.plot3D(
            eval_env.xref[:, 0],
            eval_env.xref[:, 1],
            eval_env.xref[:, 2],
            color="black",
            linewidth=2,
            linestyle="--",
            alpha=0.6,
        )
        # Add start/end dots for the reference
        ax.scatter(
            eval_env.xref[0, 0],
            eval_env.xref[0, 1],
            eval_env.xref[0, 2],
            s=100,
            alpha=0.6,
            marker="*",
        )
        # Add labels, grid, legend, and equal axis
        ax.set_xlabel("X position", fontsize=16, labelpad=15)
        ax.set_ylabel("Y position", fontsize=16, labelpad=15)
        ax.set_zlabel("Z position", fontsize=16, labelpad=15)

        # ax.view_init(elev=30, azim=-30)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # ax.zticks(fontsize=14) # zticks fontsize, if needed
        # plt.axis("equal")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        plt.savefig(
            f"fig/{eval_env.task}/tracking/{algo_name}/{algo_name}({seed})_tracking_trajectories.svg"
        )
        plt.close()

    return eval_dict, ep_norm_tracking_error


def run(args, seed, unique_id, exp_time):
    # fix seed
    seed_all(seed)

    # get env
    eval_env = call_env(args)  # always uses true dynamics
    logger, writer = setup_logger(
        args,
        unique_id,
        exp_time,
        seed,
        verbose=False,
    )

    if args.algo_name in ["lqr", "lqr-approx", "sd-lqr", "sd-lqr-approx"]:
        # make dir
        if not os.path.exists(f"model/{args.task}/{args.algo_name}"):
            os.makedirs(f"model/{args.task}/{args.algo_name}")
        # get dynamics and use it for simulation
        get_f_and_B = DynamicLearner(
            x_dim=eval_env.num_dim_x,
            action_dim=args.action_dim,
            hidden_dim=args.DynamicLearner_dim,
            Dynamic_lr=args.Dynamic_lr,
            drop_out=0.0,
            nupdates=args.dynamics_epochs,
            device=args.device,
        )
        get_f_and_B.load_state_dict(
            torch.load(f"model/{args.task}/{args.algo_name}/dynamics({seed}).pth")
        )
        get_f_and_B.eval()

        if args.algo_name in ["sd-lqr", "sd-lqr-approx"]:
            # get SDC
            SDC_func = SDCLearner(
                x_dim=eval_env.num_dim_x,
                a_dim=args.action_dim,
                hidden_dim=args.SDCLearner_dim,
                get_f_and_B=get_f_and_B,
                nupdates=args.sdc_epochs,
                device=args.device,
            )
            SDC_func.load_state_dict(
                torch.load(f"model/{args.task}/{args.algo_name}/SDC({seed}).pth")
            )
            SDC_func.eval()
        else:
            SDC_func = None
    else:
        get_f_and_B = None
        SDC_func = None

    policy = get_policy(eval_env, args, get_f_and_B, SDC_func, seed)

    eval_dict, ep_norm_tracking_error = evaluate(eval_env, args.algo_name, policy, seed)

    wandb.finish()

    return eval_dict, ep_norm_tracking_error


if __name__ == "__main__":
    # initialization
    torch.set_default_dtype(torch.float32)

    init_args = get_args()
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    random.seed(init_args.seed)
    seeds = [random.randint(1, 10_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    algo_names = [
        # "cac-approx",
        "cac-approx-deterministic",
        "cac-approx-noentropy",
        # "c3m-approx",
        # "ppo-approx",
        # "sd-lqr-approx",
        # "lqr-approx",
    ]
    tracking_error_mean_dict = {}
    tracking_error_std_dict = {}
    for algo_name in algo_names:
        dict_list = []
        tracking_error_list = []
        for seed in seeds:
            args = override_args(init_args)
            args.algo_name = algo_name
            args.seed = seed

            eval_dict, ep_norm_tracking_error = run(args, seed, unique_id, exp_time)
            dict_list.append(eval_dict)
            tracking_error_list.append(ep_norm_tracking_error)

        # pull best result
        mean_dict = {}
        ci_dict = {}

        n = len(dict_list)
        z = 1.96  # 95% confidence level for normal distribution

        for key in dict_list[0].keys():
            values = np.array([d[key] for d in dict_list])
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # sample std (ddof=1 for unbiased estimate)
            margin = z * (std / np.sqrt(n))  # margin of error

            mean_dict[key] = mean
            ci_dict[key] = margin

        print("=======================================")
        print(f"{algo_name}: Mean Results: {mean_dict[f"{algo_name}/mauc_mean"]}")
        print(
            f"{algo_name}: 95% Confidence Intervals: {ci_dict[f"{algo_name}/mauc_mean"]}"
        )

        # save mean and std dict in json
        if not os.path.exists(f"results/{args.task}/{algo_name}"):
            os.makedirs(f"results/{args.task}/{algo_name}")

        with open(f"results/{args.task}/{algo_name}/mean_dict.json", "w") as f:
            json.dump({k: v.tolist() for k, v in mean_dict.items()}, f, indent=4)
        with open(f"results/{args.task}/{algo_name}/std_dict.json", "w") as f:
            json.dump({k: v.tolist() for k, v in ci_dict.items()}, f, indent=4)

        # print tracking error
        # makesure tracking_error_list is numpy array and same length by padding
        max_length = max(len(traj) for traj in tracking_error_list)
        tracking_error_array = np.array(
            [
                np.pad(traj, (0, max_length - len(traj)), mode="edge")
                for traj in tracking_error_list
            ]
        )
        tracking_error_mean_dict[algo_name] = np.mean(tracking_error_array, axis=0)
        tracking_error_std_dict[algo_name] = (
            1.96
            * np.std(tracking_error_array, axis=0)
            / np.sqrt(len(tracking_error_list))
        )

    # draw the figure as plt.plot and fill_between
    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    # make y_label as log
    for algo_name in algo_names:
        x = np.linspace(
            0, MAX_TIME[args.task], tracking_error_mean_dict[algo_name].shape[0]
        )
        ax.plot(
            x,
            tracking_error_mean_dict[algo_name],
            label=NAMES[args.task][algo_name],
            color=COLORS[algo_name],
            linestyle=LINESTYLES[algo_name],
            linewidth=5,
            alpha=0.9,
        )
        ax.fill_between(
            x,
            tracking_error_mean_dict[algo_name] - tracking_error_std_dict[algo_name],
            tracking_error_mean_dict[algo_name] + tracking_error_std_dict[algo_name],
            alpha=0.2,
            color=COLORS[algo_name],
        )

    ax.set_xlabel("Time (s)", fontsize=30)
    ax.set_ylabel("Normalized Tracking Error", fontsize=30)

    # Set log scale
    ax.set_yscale("log")

    # Control tick font sizes (both normal and mathtext subscripts/superscripts)
    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26)

    # Force scientific/mathtext style for log ticks so subscripts scale
    # ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    # Globally adjust minor tick font size
    for label in ax.get_yminorticklabels():
        label.set_fontsize(22)

    # Legend
    ax.legend(title="mAUC", title_fontsize=28, fontsize=22, loc="best")

    # Grid and layout
    ax.grid(True, which="both", linestyle="--", linewidth=0.8)
    plt.tight_layout()

    # save figs in fig/env_name/
    if not os.path.exists(f"fig/{args.task}"):
        os.makedirs(f"fig/{args.task}")
    os.chdir(f"fig/{args.task}")

    plt.savefig(f"{args.task}_tracking_error.pdf")
    plt.savefig(f"{args.task}_tracking_error.svg")
    plt.savefig(f"{args.task}_tracking_error.png")
    plt.close()
