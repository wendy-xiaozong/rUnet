import functools
import os
from pathlib import Path
from typing import List, Optional, Tuple

import monai
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from utils.const import DIFFUSION_INPUT, DIFFUSION_LABEL
from monai.transforms import Compose
from utils.transforms import get_train_img_transforms, get_val_img_transforms, get_label_transforms
from sklearn.model_selection import train_test_split
from monai.transforms import LoadNifti, Randomizable, apply_transform
from torch.utils.data import DataLoader, Dataset


class DiffusionDataset(Dataset):
    def __init__(self, X_path: List[str], y_path: List[str], transform: Compose):
        self.X_path = X_path
        self.y_path = y_path
        self.X_transform = transform
        self.y_transform = get_label_transforms()

    def __len__(self):
        return int(len(self.X_path))

    # What is this used for?
    def randomize(self) -> None:
        MAX_SEED = np.iinfo(np.uint32).max + 1
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, i):
        self.randomize()
        loadnifti = LoadNifti()
        X_img, compatible_meta = loadnifti(self.X_path[i])
        y_img, compatible_meta = loadnifti(self.y_path[i])

        X_img = apply_transform(self.X_transform, X_img)
        y_img = apply_transform(self.y_transform, y_img)

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
        self.train_dataset = DiffusionDataset(X_path=X[:-1] * 200, y_path=y[:-1] * 200)
        self.val_dataset = DiffusionDataset(X_path=[X[-1]], y_path=[y[-1]])

    def train_dataloader(self):
        print(f"get {len(self.train_dataset)} training 3D image!")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
        )

    def val_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=8)
