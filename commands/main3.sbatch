#!/bin/bash
#SBATCH --job-name=cac-main3
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --task cartpole --project Optimal_CAC --algo-name ppo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task segway --project Optimal_CAC --algo-name ppo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task car --project Optimal_CAC --algo-name ppo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task pvtol --project Optimal_CAC --algo-name ppo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task neurallander --project Optimal_CAC --algo-name ppo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task quadrotor --project Optimal_CAC --algo-name ppo &

# === Wait for all background jobs to finish ===
wait