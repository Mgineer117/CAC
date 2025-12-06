#!/bin/bash
#SBATCH --job-name=cac-main1
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --task car --project CAC_real --algo-name c3m-approx --sample-mode Real-world &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task pvtol --project CAC_real --algo-name c3m-approx --sample-mode Real-world &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task neurallander --project CAC_real --algo-name c3m-approx --sample-mode Real-world &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task quadrotor --project CAC_real --algo-name c3m-approx --sample-mode Real-world &

CUDA_VISIBLE_DEVICES=1 python3 main.py --task car --project CAC_real --algo-name cac-approx --sample-mode Real-world &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task pvtol --project CAC_real --algo-name cac-approx --sample-mode Real-world &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task neurallander --project CAC_real --algo-name cac-approx --sample-mode Real-world &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task quadrotor --project CAC_real --algo-name cac-approx --sample-mode Real-world &

# === Wait for all background jobs to finish ===
wait