import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--project", type=str, default="Exp", help="WandB project classification"
    )
    parser.add_argument(
        "--logdir", type=str, default="log/train_log", help="name of the logging folder"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Global folder name for experiments with multiple seed tests.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Seed-specific folder name in the "group" folder.',
    )
    parser.add_argument(
        "--task",
        type=str,
        default="car",
        help="Define the task = [car, pvtol, neurallander, quadrotor].",
    )
    parser.add_argument("--algo-name", type=str, default="cac", help="Algorithm name.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of experiments for each algorithm.",
    )
    parser.add_argument(
        "--actor-lr", type=float, default=1e-4, help="Actor learning rate."
    )
    parser.add_argument(
        "--critic-lr", type=float, default=3e-4, help="Critic learning rate."
    )
    parser.add_argument(
        "--Dynamic-lr",
        type=float,
        default=1e-3,
        help="Dynamic approximator learning rate.",
    )
    parser.add_argument(
        "--SDC-lr",
        type=float,
        default=1e-3,
        help="SDC decomposition neural net learning rate.",
    )
    parser.add_argument("--W-lr", type=float, default=1e-3, help="CMG learning rate.")
    parser.add_argument(
        "--u-lr", type=float, default=1e-3, help="C3M actor learning rate."
    )
    parser.add_argument(
        "--w-ub", type=float, default=10.0, help="Contraction metric upper bound."
    )
    parser.add_argument(
        "--w-lb", type=float, default=1e-1, help="Contraction metric lower bound."
    )
    parser.add_argument(
        "--eps-clip", type=float, default=0.2, help="Epsilon clip for PPO."
    )
    parser.add_argument(
        "--eps", type=float, default=0.1, help="Used for CMG learning regularization."
    )
    parser.add_argument(
        "--lbd", type=float, default=2.0, help="Desired contraction rate."
    )
    parser.add_argument(
        "--DynamicLearner-dim",
        type=list,
        default=[256, 256],
        help="Dynamic approximator hidden layer.",
    )
    parser.add_argument(
        "--SDCLearner-dim",
        type=list,
        default=[256, 256],
        help="SDC decomposition neural net hidden layer.",
    )
    parser.add_argument(
        "--actor-dim", type=list, default=[128, 128], help="actor hidden layers."
    )
    parser.add_argument(
        "--critic-dim", type=list, default=[256, 256], help="critic hidden layers."
    )

    parser.add_argument(
        "--c3m-epochs", type=int, default=None, help="Number of training samples."
    )
    parser.add_argument(
        "--dynamics-epochs",
        type=int,
        default=10000,
        help="Number of training samples.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Number of training samples."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Number of evaluation throughout timesteps.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per reference trajectory.",
    )
    parser.add_argument(
        "--eval-num",
        type=int,
        default=10,
        help="Number of reference trajectory for evaluation.",
    )
    parser.add_argument("--sigma", type=float, default=0.0, help="Disturbance rate.")
    parser.add_argument(
        "--c3m-buffer-size", type=int, default=100_000, help="Number of mini-batches."
    )
    parser.add_argument(
        "--dynamics-buffer-size",
        type=int,
        default=2_000,
        help="Number of mini-batches.",
    )
    parser.add_argument(
        "--num-minibatch", type=int, default=4, help="Number of mini-batches."
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=2048, help="Size of each mini-batch."
    )
    parser.add_argument(
        "--K-epochs", type=int, default=5, help="Number of K epochs in PPO."
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=1e-2,
        help="PPO Target KL divergence.",
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="Generalized Advantage Estimation factor.",
    )
    parser.add_argument(
        "--entropy-scaler", type=float, default=1e-4, help="Entropy scaling factor."
    )
    parser.add_argument(
        "--W-entropy-scaler", type=float, default=1e-3, help="W entropy scaling factor."
    )
    parser.add_argument(
        "--control-scaler",
        type=float,
        default=0.0,
        help="Control scaling factor to reward.",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument(
        "--load-pretrained-model",
        action="store_true",
        help="Path to a directory for storing the log.",
    )

    parser.add_argument("--gpu-idx", type=int, default=0, help="GPU index.")

    args = parser.parse_args()
    args.device = select_device(args.gpu_idx)

    return args


def select_device(gpu_idx=0, verbose=True):
    if verbose:
        print(
            "============================================================================================"
        )
        # set device to cpu or cuda
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
    return device
