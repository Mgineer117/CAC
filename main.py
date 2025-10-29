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
from utils.get_dynamics import get_dynamics
from utils.get_sdc import get_SDC
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
    env = call_env(args)  # can use approximated dynamics
    eval_env = call_env(args)  # always uses true dynamics
    logger, writer = setup_logger(args, unique_id, exp_time, seed)

    # get dynamics and use it for simulation
    get_f_and_B, init_epochs = get_dynamics(env, args, logger, writer)
    # get SDC
    SDC_func, init_epochs = get_SDC(env, args, logger, writer, get_f_and_B, init_epochs)

    policy = get_policy(env, eval_env, args, get_f_and_B, SDC_func)

    if args.algo_name in (
        "cac",
        "cac-approx",
        "cacv2",
        "cacv2-approx",
        "ppo",
        "ppo-approx",
    ):
        sampler = OnlineSampler(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            episode_len=args.episode_len,
            batch_size=int(args.minibatch_size * args.num_minibatch),
        )

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
        trainer.train()
    elif args.algo_name in ("c3m", "c3m-approx", "c3mv2", "c3mv2-approx"):
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
    else:
        from trainer.evaluator import Evaluator

        evaluator = Evaluator(
            eval_env=eval_env,
            policy=policy,
            logger=logger,
            writer=writer,
            timesteps=args.timesteps,
            init_epochs=init_epochs,
            eval_epochs=args.log_interval,
            eval_num=args.eval_num,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
        )
        evaluator.begin_evaluate()

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
