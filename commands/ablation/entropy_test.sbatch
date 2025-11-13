#!/bin/bash
#SBATCH --job-name=cac-entropy-test
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=3
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --task cartpole --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.01 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task segway --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.01 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task car --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.01 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task pvtol --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.01 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task neurallander --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.01 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task quadrotor --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.01 &

CUDA_VISIBLE_DEVICES=0 python3 main.py --task cartpole --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.001 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task segway --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.001 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --task car --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.001 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task pvtol --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.001 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task neurallander --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.001 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task quadrotor --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.001 &

CUDA_VISIBLE_DEVICES=1 python3 main.py --task cartpole --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.0001 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task segway --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.0001 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task car --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.0001 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task pvtol --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.0001 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task neurallander --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.0001 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --task quadrotor --project Optimal_CAC_ablation --algo-name cacv2-approx --lbd 0.0 --entropy-scaler 0.0001 &

# === Wait for all background jobs to finish ===
wait