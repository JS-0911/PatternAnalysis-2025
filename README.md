#  3D Prostate MRI Segmentation — COMP3710 Final Project

**Author:** Janvhi Sharma  
**Program:** Bachelor of Engineering (Software), The University of Queensland  
**Course:** COMP3710 — Pattern Analysis  
**Model:** ImprovedUNet3D (PyTorch)  
**Best Dice:** 0.7976  
**Dataset:** HipMRI_Study_open (UQ Teaching Dataset)

---

## 1️⃣ Problem Overview

This project performs **3D prostate segmentation** on MRI scans using a custom PyTorch model — an **ImprovedUNet3D**.  
The aim was to develop a reproducible deep learning pipeline that can:
- Train efficiently on UQ’s A100 GPU cluster (Rangpur),
- Produce voxel-accurate segmentations,
- And reach a validation **Dice > 0.75** within limited compute time.

Each MRI volume is a `.nii.gz` file, and each has a corresponding label map representing the prostate region.  
The model learns to predict these 3D label maps, voxel-by-voxel.

---

## 2️ Model Architecture — ImprovedUNet3D

### Key Components
- **ResidualBlock3D:** Improves gradient flow and stability.
- **3D Encoder–Decoder (UNet-style):** Captures both local and contextual spatial patterns.
- **Skip Connections:** Preserve fine structural details between encoding and decoding paths.
- **Instance Normalisation + ReLU:** Regularises training and speeds convergence.

### Why It Works
The residual UNet design is ideal for volumetric medical data — it balances expressive power and parameter efficiency, while 3D operations maintain spatial context that 2D models would lose.

🧩 Defined in:  
`recognition/prostate3d_unet3d_jsharma/modules.py`

---

## 3️ Dataset and Pre-processing

**Source:** `/home/groups/comp3710/HipMRI_Study_open`  
- Images → `semantic_MRs/`  
- Labels → `semantic_labels_only/`

**Transformations (dataset.py):**
- `Resize3D` — scales volumes to a consistent spatial shape.  
- `Normalize3D` — z-score standardisation per channel.  
- `RandomFlip3D` + `RandomRotate3D` — light augmentations for generalisation.

Each sample outputs:
- `image`: shape (1, D, H, W)  
- `label`: one-hot (6, D, H, W)

---

## 4️ Training Details

| Setting | Value |
|----------|--------|
| Epochs | 10 |
| Batch Size | 1 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss | Dice Loss |
| Metric | Dice Coefficient |
| GPU | NVIDIA A100 |
| AMP | Enabled |
| DataLoader Workers | 4 |

**SLURM job used:** `train_job_final_10ep.slurm`

### Command
```bash
sbatch train_job_final_10ep.slurm

### Output Log (val Dice progression)

[Epoch 001] dice 0.4835
[Epoch 002] dice 0.5252
[Epoch 003] dice 0.5319
[Epoch 004] dice 0.6491
[Epoch 005] dice 0.7027
[Epoch 006] dice 0.6794
[Epoch 007] dice 0.7292
[Epoch 008] dice 0.7707
[Epoch 009] dice 0.7786
[Epoch 010] dice 0.7976
