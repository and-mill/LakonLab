#!/bin/bash --login

#SBATCH --account=fischamj_0000
#SBATCH --nodes=1
#SBATCH --partition=fat_gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=12
#SBATCH --job-name=asymflow_repa_sigma0_50k
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/mlrandwm/slurm_logs/%x_%A_%a_%j.out
#SBATCH --error=/home/mlrandwm/slurm_errors/%x_%A_%a_%j.err

set -euo pipefail

cd /home/mlrandwm/git/LakonLab
source .venv/bin/activate
export OMP_NUM_THREADS=1

# One launcher starts four workers on the GPUs allocated by Slurm.
# The config generates 50K samples and computes ADM FID online.
# Generation uses labels/noise and cached reference features; no ImageNet files needed.
torchrun --standalone --nnodes=1 --nproc_per_node=4 tools/test.py \
  configs/asymflow/asymflow_h_16_r8_repa_imagenet_test.py \
  --launcher pytorch --diff_seed \
  --cfg-options \
  model.diffusion.denoising.sigma_min=0.0 \
  work_dir="/lustre/mlrandwm/git/LakonLab/asymflow_repa_sigma0_50k"

