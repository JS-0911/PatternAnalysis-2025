#!/bin/bash
#SBATCH --job-name=pa7_debug
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/train_%x-%j.out
#SBATCH --error=logs/train_%x-%j.err

set -e
module load cuda/12.2
source ~/miniconda/etc/profile.d/conda.sh
conda activate pa2025

cd ~/comp3710/PatternAnalysis-2025/recognition/prostate3d_unet3d_jsharma

echo "=== DEBUG START ==="
hostname; date; nvidia-smi

python train.py 

echo "=== DEBUG END ==="
date
