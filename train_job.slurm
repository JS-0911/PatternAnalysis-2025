#!/bin/bash
#SBATCH --job-name=Prostate3D_FinalRun
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/train_%x-%j.out
#SBATCH --error=logs/train_%x-%j.err
#SBATCH --mail-user=s4975045@uq.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=== JOB START ==="
hostname
date
echo ">>> running train.py"

module load cuda/12.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pa2025

cd ~/comp3710/PatternAnalysis-2025/recognition/prostate3d_unet3d_jsharma

python train.py || echo "❌ train.py failed"
python predict.py || echo "❌ predict.py failed"

echo "=== JOB END ==="
date

