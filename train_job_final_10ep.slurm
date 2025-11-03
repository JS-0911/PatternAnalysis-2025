#!/bin/bash
#SBATCH --job-name=Prostate3D_Final
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%x-%j.out
#SBATCH --error=logs/train_%x-%j.err

echo "=== JOB START ==="
hostname
date
nvidia-smi

module load cuda/12.2
source ~/miniconda/etc/profile.d/conda.sh
conda activate pa2025

cd ~/comp3710/PatternAnalysis-2025/recognition/prostate3d_unet3d_jsharma || exit 1

echo ">>> Training started"
python train.py --epochs 10 || { echo "❌ train.py failed"; exit 1; }

echo ">>> Predicting after training"
python predict.py || echo "❌ predict.py failed"

echo "=== JOB END ==="
date

