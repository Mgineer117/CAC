# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #

import datetime
import random
import uuid

import torch

import wandb
from trainer.offline_trainer import C3MTrainer, DynamicsTrainer
from trainer.online_trainer import OnlineTrainer
from utils.get_args import get_args
from utils.misc import (
    concat_csv_columnwise_and_delete,
    override_args,
    seed_all,
    setup_logger,
)
from utils.sampler import OnlineSampler
from utils.utils import call_env, get_policy


def run(args, seed, unique_id, exp_time):
    # fix seed
    seed_all(seed)

    # get env
    env = call_env(args)
    eval_env = call_env(args)
    logger, writer = setup_logger(args, unique_id, exp_time, seed)

    if args.algo_name in ["cac-approx", "c3m-approx", "ppo-approx", "sd-lqr"]:
        from policy.layers.dynamic_networks import DynamicLearner

        print("[INFO] Using Dynamic Learner for dynamics approximation.")
        # learn dynamics
        Dynamic_func = DynamicLearner(
            x_dim=env.num_dim_x,
            action_dim=args.action_dim,
            hidden_dim=args.DynamicLearner_dim,
            Dynamic_lr=args.Dynamic_lr,
            drop_out=0.2,  # to prevent overfit
            device=args.device,
        )
        Dynamic_trainer = DynamicsTrainer(
            env=env,
            Dynamic_func=Dynamic_func,
            logger=logger,
            writer=writer,
            buffer_size=args.dynamics_buffer_size,
            epochs=args.dynamics_epochs,
        )
        Dynamic_trainer.train()
        init_epochs = args.dynamics_epochs
    else:
        init_epochs = 0
        Dynamic_func = None

    policy = get_policy(env, args, Dynamic_func)

    sampler = OnlineSampler(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        episode_len=args.episode_len,
        batch_size=int(args.minibatch_size * args.num_minibatch),
    )

    if args.algo_name in ["cac", "cac-approx", "ppo", "ppo-approx"]:
        trainer = OnlineTrainer(
            env=env,
            eval_env=eval_env,
            policy=policy,
            sampler=sampler,
            logger=logger,
            writer=writer,
            init_epochs=init_epochs,
            timesteps=args.timesteps,
            log_interval=args.log_interval,
            eval_num=args.eval_num,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
        )
    else:
        trainer = C3MTrainer(
            env=env,
            eval_env=eval_env,
            policy=policy,
            logger=logger,
            writer=writer,
            init_epochs=init_epochs,
            epochs=args.c3m_epochs,
            log_interval=args.log_interval,
            eval_num=args.eval_num,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
        )

    trainer.train()
    wandb.finish()


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

    for seed in seeds:
        args = override_args(init_args)
        args.seed = seed

        run(args, seed, unique_id, exp_time)
    concat_csv_columnwise_and_delete(folder_path=args.logdir)
