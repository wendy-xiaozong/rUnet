import functools
import os
from pathlib import Path

from monai import transforms
from typing import List, Optional, Tuple

import monai
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from utils.const import DIFFUSION_INPUT, DIFFUSION_LABEL
from monai.transforms import Compose
from utils.transforms import get_diffusion_preprocess
from sklearn.model_selection import train_test_split
from monai.transforms import LoadNifti, Randomizable, apply_transform
from torch.utils.data import DataLoader, Dataset


class DiffusionDataset(Dataset):
    def __init__(self, X_path: List[str], y_path: List[str], transform: Compose):
        self.X_path = X_path
        self.y_path = y_path
        self.transform = transforms

    def __len__(self):
        return int(len(self.X_path))

    def __getitem__(self, i):
        self.randomize()
        loadnifti = LoadNifti()
        X_img, compatible_meta = loadnifti(self.X_path[i])
        y_img, compatible_meta = loadnifti(self.y_path[i])

        X_img = apply_transform(self.transform, X_img)
        y_img = apply_transform(self.transform, y_img)

        return X_img, y_img


class DataModule_Diffusion(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    # perform on every GPU
    def setup(self, stage: Optional[str] = None) -> None:
        X = sorted(list(DIFFUSION_INPUT.glob("**/*.nii")))
        y = sorted(list(DIFFUSION_LABEL.glob("**/*.nii")))

        # train_transforms = get_train_img_transforms()
        # val_transforms = get_val_img_transforms()
        preprocess = get_diffusion_preprocess()

        self.train_dataset = DiffusionDataset(X_path=X[:-1] * 200, y_path=y[:-1] * 200, transform=preprocess)
        self.val_dataset = DiffusionDataset(X_path=[X[-1]] * 4, y_path=[y[-1]] * 4, transform=preprocess)

    def train_dataloader(self):
        print(f"get {len(self.train_dataset)} training 3D image!")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=8)