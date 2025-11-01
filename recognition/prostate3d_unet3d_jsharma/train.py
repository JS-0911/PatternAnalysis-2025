# Commit milestone: refined training and validation loops, added Dice loss monitoring and seed reproducibility
"""
train.py — Train/validate/save the 3D model, log metrics & plots.

Spec alignment:
- imports model from modules.py and dataset/augmentations from dataset.py
- logs loss + Dice, saves best checkpoint, saves training curves
- provides CLI args so Rangpur/Colab paths are easy to pass
"""

from __future__ import annotations
import os
import argparse
import random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .modules import ImprovedUNet3D
from .dataset import (
    Prostate3DDataset, Resize3D, Normalize3D, RandomFlip3D, RandomRotate3D
)


# ------------------------- utilities -------------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_coef_onehot(pred_logits: torch.Tensor, target_oh: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_logits: (N,C,D,H,W) raw scores
    target_oh:   (N,C,D,H,W) one-hot labels
    returns per-class dice averaged across batch & voxels (Tensor[C])
    """
    probs = F.softmax(pred_logits, dim=1)
    probs = probs.flatten(2)            # (N,C,V)
    target = target_oh.flatten(2)       # (N,C,V)
    inter = (probs * target).sum(dim=(0, 2))
    denom = probs.sum(dim=(0, 2)) + target.sum(dim=(0, 2))
    dice_c = (2 * inter + eps) / (denom + eps)
    return dice_c  # (C,)


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target_oh: torch.Tensor) -> torch.Tensor:
        dice_c = dice_coef_onehot(logits, target_oh, self.eps)
        return 1.0 - dice_c.mean()


# ------------------------- training -------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    epoch_loss, epoch_dice = 0.0, 0.0
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, lbls)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dice_c = dice_coef_onehot(logits, lbls).mean().item()
        epoch_loss += float(loss.item())
        epoch_dice += dice_c

    n = len(loader)
    return epoch_loss / n, epoch_dice / n


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    val_loss, val_dice = 0.0, 0.0
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, lbls)
        dice_c = dice_coef_onehot(logits, lbls).mean().item()
        val_loss += float(loss.item())
        val_dice += dice_c
    n = len(loader)
    return val_loss / n, val_dice / n


def build_loaders(
    images_dir: str,
    labels_dir: str,
    num_classes: int,
    out_size=(96, 128, 128),
    batch_size: int = 2,
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # transforms
    common = [
        Resize3D(out_size),
        Normalize3D(),
    ]
    aug = [RandomFlip3D(0.5), RandomRotate3D(10.0, 0.5)]

    full = Prostate3DDataset(
        images_dir, labels_dir, num_classes=num_classes,
        transform=common + aug
    )

    # split indices
    n = len(full)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test
    train_set, val_set, test_set = torch.utils.data.random_split(
        full, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    # IMPORTANT: turn off augmentation for val/test by re-wrapping without aug
    def wrap_no_aug(subset):
        base = subset.dataset
        return torch.utils.data.Subset(
            Prostate3DDataset(
                base.images_dir, base.labels_dir, num_classes=num_classes,
                transform=common  # no aug
            ),
            subset.indices
        )

    val_set = wrap_no_aug(val_set)
    test_set = wrap_no_aug(test_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def plot_curves(history: Dict[str, list], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # loss
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout(); plt.savefig(out_dir / "loss.png"); plt.close()

    # dice
    plt.figure()
    plt.plot(history["train_dice"], label="train")
    plt.plot(history["val_dice"], label="val")
    plt.xlabel("epoch"); plt.ylabel("mean Dice"); plt.legend(); plt.title("Dice")
    plt.tight_layout(); plt.savefig(out_dir / "dice.png"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, default="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs")
    ap.add_argument("--labels_dir", type=str, default="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only")
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="recognition/prostate3d_unet3d_jsharma/outputs")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_loader, val_loader, test_loader = build_loaders(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # model/opt/loss
    model = ImprovedUNet3D(in_channels=1, num_classes=args.num_classes).to(device)
    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = DiceLoss()

    out_dir = Path(args.out)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # train
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_val = -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_dice = validate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice); history["val_dice"].append(va_dice)

        print(f"[Epoch {ep:03d}] loss {tr_loss:.4f}/{va_loss:.4f}  dice {tr_dice:.4f}/{va_dice:.4f}")

        # save best
        if va_dice > best_val:
            best_val = va_dice
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    plot_curves(history, out_dir)

    # also save final
    torch.save(model.state_dict(), ckpt_dir / "last.pt")
    print(f"✔ training complete. best val dice: {best_val:.4f}. checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
