# predict.py
# Author: Janvhi Sharma (COMP3710 Project 7)
# Inference and evaluation script for 3D UNet segmentation

import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from dataset import Prostate3DDataset
from modules import UNet3D, DiceLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = "/home/groups/comp3710/PROSTATE_3D"
CKPT_PATH = "best_model.ckpt"
OUT_DIR = "predictions"
os.makedirs(OUT_DIR, exist_ok=True)


def dice_score(preds, targets):
    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)
    intersection = (preds * targets).float().sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + 1e-8)


def save_slice(volume, path):
    """Save a middle axial slice as PNG for visualization."""
    import matplotlib.pyplot as plt
    mid = volume.shape[1] // 2
    plt.imshow(volume[0, mid, :, :], cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    print(f"Running inference on device: {DEVICE}")
    model = UNet3D().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    test_ds = Prostate3DDataset(ROOT_DIR, split="test")
    test_loader = DataLoader(test_ds, batch_size=1)

    dice_total = 0
    for idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            preds = model(x)
        dice = dice_score(preds, y)
        dice_total += dice.item()

        # save example slice
        pred_np = torch.argmax(preds, dim=1).cpu().numpy()[0]
        save_slice(pred_np[np.newaxis, ...],
                   os.path.join(OUT_DIR, f"sample_{idx}.png"))

        print(f"Sample {idx} → Dice: {dice:.4f}")

    mean_dice = dice_total / len(test_loader)
    print(f"\n Average Dice on test set: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
