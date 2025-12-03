#!/bin/bash
#SBATCH --job-name=cac-main3
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --time=2-00:00:00
#SBATCH --output=cac.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# Load conda
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate cac

# === Run CACeriments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --project Exp-c3m --task car --algo-name c3m --lbd 0.75 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project Exp-c3m --task pvtol --algo-name c3m --lbd 0.75 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project Exp-c3m --task neurallander --algo-name c3m --lbd 0.75 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project Exp-c3m --task quadrotor --algo-name c3m --lbd 0.75 &

CUDA_VISIBLE_DEVICES=1 python3 main.py --project Exp-c3m --task car --algo-name c3m --lbd 1.0 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --project Exp-c3m --task pvtol --algo-name c3m --lbd 1.0 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --project Exp-c3m --task neurallander --algo-name c3m --lbd 1.0 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --project Exp-c3m --task quadrotor --algo-name c3m --lbd 1.0 &

CUDA_VISIBLE_DEVICES=2 python3 main.py --project Exp-c3m --task car --algo-name c3m --lbd 1.5 &
CUDA_VISIBLE_DEVICES=2 python3 main.py --project Exp-c3m --task pvtol --algo-name c3m --lbd 1.5 &
CUDA_VISIBLE_DEVICES=2 python3 main.py --project Exp-c3m --task neurallander --algo-name c3m --lbd 1.5 &
CUDA_VISIBLE_DEVICES=2 python3 main.py --project Exp-c3m --task quadrotor --algo-name c3m --lbd 1.5 &

# === Wait for all background jobs to finish ===
wait