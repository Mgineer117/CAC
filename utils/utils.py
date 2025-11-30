from csv import writer

import numpy as np
import torch
import torch.nn as nn

from envs import (
    CarEnv,
    CartPoleEnv,
    FlapperEnv,
    NeuralLanderEnv,
    PvtolEnv,
    QuadRotorEnv,
    SegwayEnv,
    TurtlebotEnv,
)


def call_env(args):
    # 1. Define the map of strings to Class objects
    env_map = {
        "car": CarEnv,
        "pvtol": PvtolEnv,
        "quadrotor": QuadRotorEnv,
        "neurallander": NeuralLanderEnv,
        "segway": SegwayEnv,
        "turtlebot": TurtlebotEnv,
        "cartpole": CartPoleEnv,
        "flapper": FlapperEnv,
    }

    # 2. Check existence
    if args.task not in env_map:
        raise NotImplementedError(f"{args.task} is not implemented.")

    # 3. Instantiate once using the common arguments
    env = env_map[args.task](sample_mode=args.sample_mode, reward_mode=args.reward_mode)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.episode_len = env.episode_len

    return env


def get_policy(env, eval_env, args, get_f_and_B, SDC_func=None):
    algo_name = args.algo_name

    if algo_name.startswith(("lqr", "sd-lqr")):
        from policy.lqr import LQR
        from policy.sd_lqr import SD_LQR

        if algo_name.startswith("lqr"):
            policy = LQR(
                x_dim=env.num_dim_x,
                action_dim=args.action_dim,
                get_f_and_B=get_f_and_B,
            )
        elif algo_name.startswith("sd-lqr"):
            policy = SD_LQR(
                x_dim=env.num_dim_x,
                action_dim=args.action_dim,
                get_f_and_B=get_f_and_B,
                SDC_func=SDC_func,
            )

    elif algo_name.startswith("ppo"):
        from policy.layers.c3m_networks import C3M_U_Gaussian
        from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
        from policy.ppo import PPO

        nupdates = args.timesteps / (args.minibatch_size * args.num_minibatch)

        actor = C3M_U_Gaussian(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            task=args.task,
        )
        critic = PPO_Critic(args.state_dim, hidden_dim=args.critic_dim)

        policy = PPO(
            x_dim=env.num_dim_x,
            actor=actor,
            critic=critic,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            eps_clip=args.eps_clip,
            entropy_scaler=args.entropy_scaler,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae=args.gae,
            K=args.K_epochs,
            nupdates=nupdates,
            device=args.device,
        )

    elif algo_name.startswith("c3m"):
        from policy.c3m import C3M
        from policy.c3mv2 import C3Mv2
        from policy.c3mv3 import C3Mv3
        from policy.layers.c3m_networks import C3M_U, C3M_W, C3M_U_Gaussian

        W_func = C3M_W(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            task=args.task,
            hidden_dim=[128, 128],
            activation=nn.Tanh(),
            device=args.device,
        )
        u_func = C3M_U(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            task=args.task,
        )

        data = env.get_rollout(args.c3m_buffer_size, mode="c3m")
        C3M_class = (
            C3Mv3
            if algo_name.startswith("c3mv3")
            else C3Mv2 if algo_name.startswith("c3mv2") else C3M
        )

        policy = C3M_class(
            x_dim=env.num_dim_x,
            action_dim=args.action_dim,
            W_func=W_func,
            u_func=u_func,
            data=data,
            get_f_and_B=get_f_and_B,
            W_lr=args.W_lr,
            u_lr=args.u_lr,
            lbd=args.lbd,
            eps=args.eps,
            w_ub=args.w_ub,
            w_lb=args.w_lb,
            gamma=args.gamma,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            nupdates=args.c3m_epochs,
            device=args.device,
        )

    elif algo_name.startswith("cac"):
        from policy.cac import CAC
        from policy.cacv2 import CACv2
        from policy.cacv3 import CACv3
        from policy.layers.c3m_networks import C3M_W, C3M_U_Gaussian, C3M_W_Gaussian
        from policy.layers.ppo_networks import PPO_Actor, PPO_Critic

        nupdates = args.timesteps / (args.minibatch_size * args.num_minibatch)

        W_func = C3M_W_Gaussian(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            hidden_dim=[128, 128],
            activation=nn.Tanh(),
            device=args.device,
        )

        actor = C3M_U_Gaussian(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            task=args.task,
        )

        critic = PPO_Critic(args.state_dim, hidden_dim=args.critic_dim)

        data = env.get_rollout(args.c3m_buffer_size, mode="c3m")
        policy_class = (
            CACv3
            if algo_name.startswith("cacv3")
            else CACv2 if algo_name.startswith("cacv2") else CAC
        )

        policy = policy_class(
            x_dim=env.num_dim_x,
            W_func=W_func,
            get_f_and_B=get_f_and_B,
            data=data,
            actor=actor,
            critic=critic,
            W_lr=args.W_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            w_ub=args.w_ub,
            w_lb=args.w_lb,
            lbd=args.lbd,
            eps=args.eps,
            eps_clip=args.eps_clip,
            W_entropy_scaler=args.W_entropy_scaler,
            entropy_scaler=args.entropy_scaler,
            tracking_scaler=env.tracking_scaler,
            control_scaler=env.control_scaler,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae=args.gae,
            K=args.K_epochs,
            nupdates=nupdates,
            device=args.device,
        )

    return policy
