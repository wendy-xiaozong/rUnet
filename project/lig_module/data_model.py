import functools
import os
from pathlib import Path
from typing import List, Optional, Tuple

import monai
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from utils.const import DATA_ROOT
from monai.transforms import Compose
from utils.transforms import get_train_img_transforms, get_val_img_transforms, get_label_transforms
from sklearn.model_selection import train_test_split
from monai.transforms import LoadNifti, Randomizable, apply_transform
from torch.utils.data import DataLoader, Dataset


class BraTSDataset(Dataset, Randomizable):
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
        X_t1_img, compatible_meta_t1 = loadnifti(self.X_path[i])
        y_t2_img, compatible_meta_t2 = loadnifti(self.y_path[i])

        if isinstance(self.X_transform, Randomizable):
            self.X_transform.set_random_state(seed=self._seed)
            self.y_transform.set_random_state(seed=self._seed)
        X_t1_img = apply_transform(self.X_transform, X_t1_img)
        y_t2_img = apply_transform(self.y_transform, y_t2_img)

        return X_t1_img, y_t2_img


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    # perform on every GPU
    def setup(self, stage: Optional[str] = None) -> None:
        X_t1 = [sorted(list(DATA_ROOT.glob("**/*t1.nii.gz")))[0]] * 1000
        y_t2 = [sorted(list(DATA_ROOT.glob("**/*t2.nii.gz")))[0]] * 1000

        # random_state = random.randint(0, 100)
        # X_train, X_val, y_train, y_val = train_test_split(X_t1, y_t2, test_size=0.2, random_state=random_state)

        train_transforms = get_train_img_transforms()
        val_transforms = get_val_img_transforms()
        # self.train_dataset = BraTSDataset(X_path=X_train, y_path=y_train, transform=train_transforms)
        # self.val_dataset = BraTSDataset(X_path=X_val, y_path=y_val, transform=val_transforms)

        self.train_dataset = BraTSDataset(X_path=X_t1[:-4], y_path=y_t2[:-4], transform=train_transforms)
        self.val_dataset = BraTSDataset(X_path=X_t1[-4:], y_path=y_t2[-4:], transform=val_transforms)

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
