#!/bin/bash
#SBATCH --job-name=cac
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
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

# === Run cac_neweriments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --task car --project cac_new --algo-name ppo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --task pvtol --project cac_new --algo-name ppo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --task neurallander --project cac_new --algo-name ppo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --task quadrotor --project cac_new --algo-name ppo & 

CUDA_VISIBLE_DEVICES=0 python3 main.py --task car --project cac_new --algo-name cac-approx & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --task pvtol --project cac_new --algo-name cac-approx & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --task neurallander --project cac_new --algo-name cac-approx & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --task quadrotor --project cac_new --algo-name cac-approx &

CUDA_VISIBLE_DEVICES=0 python3 main.py --task car --project cac_new --algo-name c3m-approx & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --task pvtol --project cac_new --algo-name c3m-approx & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task neurallander --project cac_new --algo-name c3m-approx & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task quadrotor --project cac_new --algo-name c3m-approx & 

CUDA_VISIBLE_DEVICES=1 python3 main.py --task car --project cac_new --algo-name lqr-approx & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task pvtol --project cac_new --algo-name lqr-approx & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task neurallander --project cac_new --algo-name lqr-approx & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task quadrotor --project cac_new --algo-name lqr-approx & 

CUDA_VISIBLE_DEVICES=1 python3 main.py --task car --project cac_new --algo-name sd-lqr & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task pvtol --project cac_new --algo-name sd-lqr & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task neurallander --project cac_new --algo-name sd-lqr & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task quadrotor --project cac_new --algo-name sd-lqr & 

# === Wait for all background jobs to finish ===
wait