# dataset.py
# Author: Janvhi Sharma (COMP3710 Project 7)
# Dataset loader for 3D Prostate MRI segmentation

import os
from glob import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def normalize(volume):
    volume = volume.astype(np.float32)
    p1, p99 = np.percentile(volume, (1, 99))
    volume = np.clip(volume, p1, p99)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume


class Prostate3DDataset(Dataset):
    def __init__(self, root_dir, split='train', resize=(96, 128, 128), n_classes=2):
        img_dir = os.path.join(root_dir, split, 'images')
        lbl_dir = os.path.join(root_dir, split, 'labels')

        self.img_paths = sorted(glob(os.path.join(img_dir, '*.nii*')))
        self.lbl_paths = sorted(glob(os.path.join(lbl_dir, '*.nii*')))
        self.resize = resize
        self.n_classes = n_classes

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = nib.load(self.img_paths[idx]).get_fdata()
        lbl = nib.load(self.lbl_paths[idx]).get_fdata()

        img = normalize(img)
        lbl = (lbl > 0).astype(np.int64)

        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        lbl = torch.tensor(lbl).unsqueeze(0)

        img = F.interpolate(img, size=self.resize, mode='trilinear', align_corners=False).squeeze(0)
        lbl = F.interpolate(lbl.float().unsqueeze(0), size=self.resize, mode='nearest').squeeze(0)

        one_hot = F.one_hot(lbl.long(), num_classes=self.n_classes).permute(3, 0, 1, 2).float()

        return img, one_hot
