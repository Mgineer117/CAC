#!/bin/bash
#SBATCH --job-name=cac
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:4
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --task car --project CAC --algo-name ppo-approx & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --task pvtol --project CAC --algo-name ppo-approx & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --task neurallander --project CAC --algo-name ppo-approx & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --task quadrotor --project CAC --algo-name ppo-approx & 

CUDA_VISIBLE_DEVICES=1 python3 main.py --task car --project CAC --algo-name cac-approx & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task pvtol --project CAC --algo-name cac-approx & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task neurallander --project CAC --algo-name cac-approx & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --task quadrotor --project CAC --algo-name cac-approx &

CUDA_VISIBLE_DEVICES=2 python3 main.py --task car --project CAC --algo-name c3m-approx & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --task pvtol --project CAC --algo-name c3m-approx & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --task neurallander --project CAC --algo-name c3m-approx & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --task quadrotor --project CAC --algo-name c3m-approx & 

CUDA_VISIBLE_DEVICES=3 python3 main.py --task car --project CAC --algo-name lqr-approx & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --task pvtol --project CAC --algo-name lqr-approx & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --task neurallander --project CAC --algo-name lqr-approx & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --task quadrotor --project CAC --algo-name lqr-approx & 

CUDA_VISIBLE_DEVICES=3 python3 main.py --task car --project CAC --algo-name sd-lqr-approx & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --task pvtol --project CAC --algo-name sd-lqr-approx & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --task neurallander --project CAC --algo-name sd-lqr-approx & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --task quadrotor --project CAC --algo-name sd-lqr-approx & 

# === Wait for all background jobs to finish ===
wait