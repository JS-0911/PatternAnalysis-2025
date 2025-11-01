# Commit milestone: integrated evaluation loop and Dice metric computation for final testing phase
"""
predict.py — Load a trained checkpoint and report Dice on a held-out set.
Also demonstrates saving a few predicted masks as NIfTI for the README.

Usage example (Rangpur/Colab):
  python -m recognition.prostate3d_unet3d_jsharma.predict \
    --images_dir /home/groups/comp3710/HipMRI_Study_open/semantic_MRs \
    --labels_dir /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only \
    --ckpt recognition/prostate3d_unet3d_jsharma/outputs/checkpoints/best.pt
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .modules import ImprovedUNet3D
from .dataset import Prostate3DDataset, Resize3D, Normalize3D


@torch.no_grad()
def per_class_dice(logits: torch.Tensor, target_oh: torch.Tensor, eps: float = 1e-6):
    probs = F.softmax(logits, dim=1)
    probs = probs.flatten(2)            # (N,C,V)
    target = target_oh.flatten(2)
    inter = (probs * target).sum(dim=(0, 2))
    denom = probs.sum(dim=(0, 2)) + target.sum(dim=(0, 2))
    dice_c = (2 * inter + eps) / (denom + eps)
    return dice_c  # (C,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--labels_dir", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="recognition/prostate3d_unet3d_jsharma/outputs/predicts")
    ap.add_argument("--n_save", type=int, default=3, help="how many volumes to save as NIfTI")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # deterministic val/test pipeline
    ds = Prostate3DDataset(
        args.images_dir, args.labels_dir, num_classes=args.num_classes,
        transform=[Resize3D((96, 128, 128)), Normalize3D()]
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = ImprovedUNet3D(in_channels=1, num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    dice_sum = torch.zeros(args.num_classes, device=device)
    count = 0

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, lbl_oh) in enumerate(loader):
        img = img.to(device); lbl_oh = lbl_oh.to(device)
        logits = model(img)

        dice_c = per_class_dice(logits, lbl_oh)
        dice_sum += dice_c
        count += 1

        # save a few predictions as NIfTI for the README
        if i < args.n_save:
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)  # (D,H,W)
            # save with an identity affine (no physical coords needed for report)
            nif = nib.Nifti1Image(pred, affine=np.eye(4))
            nib.save(nif, out_dir / f"pred_{i:03d}.nii.gz")

    mean_per_class = (dice_sum / count).cpu().numpy()
    print("Per-class Dice:", np.round(mean_per_class, 4).tolist())
    print("Mean Dice:", float(np.mean(mean_per_class)))


if __name__ == "__main__":
    main()
