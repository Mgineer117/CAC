from csv import writer

import numpy as np
import torch
import torch.nn as nn

from envs.car import CarEnv
from envs.neurallander import NeuralLanderEnv
from envs.pvtol import PvtolEnv
from envs.quadrotor import QuadRotorEnv


def call_env(args):
    task = args.task

    if task == "car":
        env = CarEnv(sigma=args.sigma)
    elif task == "pvtol":
        env = PvtolEnv(sigma=args.sigma)
    elif task == "quadrotor":
        env = QuadRotorEnv(sigma=args.sigma)
    elif task == "neurallander":
        env = NeuralLanderEnv(sigma=args.sigma)
    else:
        raise NotImplementedError(f"{task} is not implemented.")

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.episode_len = env.episode_len

    return env


def get_policy(env, eval_env, args, get_f_and_B, SDC_func=None):
    algo_name = args.algo_name

    if algo_name in ("lqr", "lqr-approx", "sd-lqr", "sd-lqr-approx"):
        from policy.lqr import LQR
        from policy.sd_lqr import SD_LQR

        if algo_name in ("lqr", "lqr-approx"):
            policy = LQR(
                x_dim=env.num_dim_x,
                action_dim=args.action_dim,
                get_f_and_B=get_f_and_B,
            )
        elif algo_name in ("sd-lqr", "sd-lqr-approx"):
            policy = SD_LQR(
                x_dim=env.num_dim_x,
                action_dim=args.action_dim,
                get_f_and_B=get_f_and_B,
                SDC_func=SDC_func,
            )

    elif algo_name in ("ppo", "ppo-approx"):
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

    elif algo_name in ("c3m", "c3m-approx"):
        from policy.c3m import C3M
        from policy.layers.c3m_networks import C3M_U, C3M_W

        W_func = C3M_W(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            w_lb=args.w_lb,
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
        policy = C3M(
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
            nupdates=args.c3m_epochs,
            device=args.device,
        )

    elif algo_name in ("cac", "cac-approx"):
        from policy.cac import CAC
        from policy.layers.c3m_networks import C3M_W, C3M_U_Gaussian, C3M_W_Gaussian
        from policy.layers.ppo_networks import PPO_Actor, PPO_Critic

        nupdates = args.timesteps / (args.minibatch_size * args.num_minibatch)

        # W_func = C3M_W_Gaussian(
        #     x_dim=env.num_dim_x,
        #     state_dim=args.state_dim,
        #     hidden_dim=[128, 128],
        #     w_lb=args.w_lb,
        #     activation=nn.Tanh(),
        #     device=args.device,
        # )
        W_func = C3M_W(
            x_dim=env.num_dim_x,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            w_lb=args.w_lb,
            task=args.task,
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

        # actor = PPO_Actor(args.state_dim - 1, args.actor_dim, args.action_dim)

        critic = PPO_Critic(args.state_dim, hidden_dim=args.critic_dim)

        data = env.get_rollout(args.c3m_buffer_size, mode="c3m")
        policy = CAC(
            x_dim=env.num_dim_x,
            W_func=W_func,
            get_f_and_B=get_f_and_B,
            true_get_f_and_B=eval_env.get_f_and_B,
            data=data,
            actor=actor,
            critic=critic,
            W_lr=args.W_lr,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            num_minibatch=args.num_minibatch,
            minibatch_size=args.minibatch_size,
            w_ub=args.w_ub,
            lbd=args.lbd,
            eps=args.eps,
            eps_clip=args.eps_clip,
            W_entropy_scaler=args.W_entropy_scaler,
            entropy_scaler=args.entropy_scaler,
            control_scaler=args.control_scaler,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae=args.gae,
            K=args.K_epochs,
            nupdates=nupdates,
            dt=env.dt,
            device=args.device,
        )

    return policy
