# train.py
# Author: Janvhi Sharma (COMP3710 Project 7)
# Training script for 3D UNet segmentation on Prostate dataset

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Prostate3DDataset
from modules import UNet3D, DiceLoss


# ---------- CONFIGURATION ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = "/home/groups/comp3710/PROSTATE_3D"  # Update if needed
SAVE_PATH = "best_model.ckpt"
EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-3
PRINT_FREQ = 5


# ---------- METRIC FUNCTION ----------
def dice_score(preds, targets):
    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)
    intersection = (preds * targets).float().sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + 1e-8)


# ---------- TRAINING LOOP ----------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for step, (x, y) in enumerate(tqdm(loader, desc="Training")):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------- VALIDATION LOOP ----------
def validate(model, loader):
    model.eval()
    dice_total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            dice_total += dice_score(preds, y)
    return dice_total / len(loader)


# ---------- MAIN ----------
def main():
    print(f"Using device: {DEVICE}")
    train_ds = Prostate3DDataset(ROOT_DIR, split="train")
    val_ds = Prostate3DDataset(ROOT_DIR, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = UNet3D().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = DiceLoss()

    best_dice = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] ----------------------")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_dice = validate(model, val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Dice = {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f" New best model saved with Dice = {best_dice:.4f}")

    print(f"\nTraining completed. Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
