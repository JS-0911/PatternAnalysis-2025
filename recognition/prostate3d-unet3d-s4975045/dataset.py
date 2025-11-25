# Commit milestone: dataset loader finalized
# Commit milestone: verified dataset loader behaviour (ready for submission)

"""
dataset.py — Data loading & preprocessing for 3D prostate MRI

Implements:
- Prostate3DDataset: loads NIfTI volumes (.nii / .nii.gz) for images and labels
- Light 3D transforms: Resize3D, Normalize3D, RandomFlip3D, RandomRotate3D
- Utility: one_hot_3d() to convert integer masks -> one-hot channel format

Returns tensors shaped for PyTorch 3D models:
  image: (C=1, D, H, W)          # single-channel MRI volume
  label: (C=num_classes, D, H, W) # per-class one-hot mask
"""

from __future__ import annotations
import os
import math
import random
from typing import Iterable, Tuple, Dict

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, rotate

import torch
from torch.utils.data import Dataset


# ------------------------- helpers -------------------------

def one_hot_3d(mask_3d: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert an integer mask (D,H,W) with values [0..num_classes-1]
    into one-hot channels (C,D,H,W).
    """
    mask_3d = mask_3d.astype(np.int64)
    oh = np.zeros((num_classes, *mask_3d.shape), dtype=np.float32)
    for c in range(num_classes):
        oh[c] = (mask_3d == c).astype(np.float32)
    return oh


def load_nifti(path: str) -> np.ndarray:
    """
    Load a NIfTI volume and return a float32 numpy array.
    If image has an extra 4th dim (e.g., shape D,H,W,1), squeeze it.
    """
    vol = nib.load(path).get_fdata(caching="unchanged")
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = np.squeeze(vol, axis=-1)
    return vol  # shape (D,H,W)


# ------------------------- transforms -------------------------

class Resize3D:
    """Resize a (C,D,H,W) or (D,H,W) volume to target (D,H,W) using scipy.zoom."""
    def __init__(self, out_size: Tuple[int, int, int]):
        self.out_size = out_size  # (D,H,W)

    def _zoom(self, vol: np.ndarray, order: int) -> np.ndarray:
        if vol.ndim == 4:  # (C,D,H,W) -> zoom per channel
            c, d, h, w = vol.shape
            zd, zh, zw = self.out_size[0] / d, self.out_size[1] / h, self.out_size[2] / w
            out = np.zeros((c, *self.out_size), dtype=vol.dtype)
            for i in range(c):
                out[i] = zoom(vol[i], (zd, zh, zw), order=order)
            return out
        else:  # (D,H,W)
            d, h, w = vol.shape
            zd, zh, zw = self.out_size[0] / d, self.out_size[1] / h, self.out_size[2] / w
            return zoom(vol, (zd, zh, zw), order=order)

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        img, lbl = sample["image"], sample["label"]
        # image: linear (order=1), label (nearest/0) to preserve classes
        return {"image": self._zoom(img, 1), "label": self._zoom(lbl, 0)}


class Normalize3D:
    """Z-score normalize image (channel-wise). label is returned unchanged."""
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        img, lbl = sample["image"], sample["label"]
        if img.ndim == 4:
            # per-channel normalize
            for c in range(img.shape[0]):
                mu, sd = float(np.mean(img[c])), float(np.std(img[c]) + 1e-6)
                img[c] = (img[c] - mu) / sd
        else:
            mu, sd = float(np.mean(img)), float(np.std(img) + 1e-6)
            img = (img - mu) / sd
        return {"image": img, "label": lbl}


class RandomFlip3D:
    """Random mirror flips along D/H/W axes with p=0.5 each."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        img, lbl = sample["image"], sample["label"]
        # work on (C,D,H,W)
        if img.ndim == 3:  # (D,H,W) -> add channel
            img = img[None, ...]
        axes = [1, 2, 3]  # D,H,W indices
        for ax in axes:
            if random.random() < self.p:
                img = np.flip(img, axis=ax).copy()
                lbl = np.flip(lbl, axis=ax).copy()
        return {"image": img, "label": lbl}


class RandomRotate3D:
    """
    Small random rotations (in degrees) around each plane.
    Uses order=1 (img) and order=0 (lbl). Keeps shape.
    """
    def __init__(self, max_deg: float = 10.0, p: float = 0.5):
        self.max_deg = max_deg
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if random.random() >= self.p:
            return sample
        img, lbl = sample["image"], sample["label"]
        if img.ndim == 3:
            img = img[None, ...]
        # rotate around (D,H), (H,W), (D,W)
        for axes in [(0, 1), (1, 2), (0, 2)]:
            ang = random.uniform(-self.max_deg, self.max_deg)
            # image: order=1, label: nearest (0)
            for c in range(img.shape[0]):
                img[c] = rotate(img[c], ang, axes=axes, reshape=False, order=1, mode="nearest")
            for c in range(lbl.shape[0]):
                lbl[c] = rotate(lbl[c], ang, axes=axes, reshape=False, order=0, mode="nearest")
        return {"image": img, "label": lbl}


# ------------------------- dataset -------------------------

class Prostate3DDataset(Dataset):
    """
    Generic dataset reading:
      images_dir: folder with MRI volumes (NIfTI)
      labels_dir: folder with integer masks (same filenames, or map via label_suffix)
    Produces:
      image tensor (1,D,H,W), label tensor (C,D,H,W) one-hot
    """
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        num_classes: int = 6,
        label_suffix: str = "",
        transform: Iterable = (),
        file_exts: Tuple[str, ...] = (".nii", ".nii.gz"),
    ):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.num_classes = int(num_classes)
        self.label_suffix = label_suffix
        self.transform = list(transform) if transform else []
        # gather filenames
        self.img_names = sorted([f for f in os.listdir(images_dir) if f.endswith(file_exts)])
        if len(self.img_names) == 0:
            raise FileNotFoundError(f"No NIfTI files found in: {images_dir}")

    def __len__(self) -> int:
        return len(self.img_names)

    def _label_name_for(self, img_name: str) -> str:
        """
        If labels have a suffix or live in another folder with mirrored names,
        adapt here. By default, uses same basename + label_suffix.
        """
        base = img_name.replace(".nii.gz", "").replace(".nii", "")
        base = base.replace("LFOV", "SEMANTIC")
        if self.label_suffix:
            base = f"{base}{self.label_suffix}"
        # assume .nii.gz if exists, else .nii
        for ext in (".nii.gz", ".nii"):
            candidate = os.path.join(self.labels_dir, base + ext)
            if os.path.exists(candidate):
                return candidate
        # fallback to same name in labels_dir
        return os.path.join(self.labels_dir, img_name)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.img_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        lbl_path = self._label_name_for(img_name)

        img = load_nifti(img_path)            # (D,H,W) float32
        lbl_int = load_nifti(lbl_path)        # (D,H,W) int (0..C-1)
        lbl_int = np.rint(lbl_int).astype(np.int64)  # ensure integer classes

        # shape to (C,D,H,W)
        img = img[None, ...]                       # (1,D,H,W)
        lbl = one_hot_3d(lbl_int, self.num_classes)  # (C,D,H,W)

        sample = {"image": img, "label": lbl}
        for t in self.transform:
            sample = t(sample)

        # numpy -> torch
        img_t = torch.from_numpy(sample["image"]).float()
        lbl_t = torch.from_numpy(sample["label"]).float()
        return img_t, lbl_t
