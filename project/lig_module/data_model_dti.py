import functools
import os
from project.utils.const import DATA_ROOT
import torch
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
from utils.transforms import get_diffusion_preprocess, get_diffusion_label_preprocess
from sklearn.model_selection import train_test_split
from monai.transforms import LoadNifti, apply_transform
from torch.utils.data import DataLoader, Dataset


class DiffusionDataset(Dataset):
    def __init__(self, path: List[str], X_transform: Compose):
        self.path = path
        self.X_transform = X_transform
        self.y_transform = get_diffusion_label_preprocess()

    def __len__(self):
        return int(len(self.path))

    def __getitem__(self, i):
        tmp = np.load(self.path[i])
        X_img, y_img = tmp["X"], tmp["y"]

        return torch.from_numpy(X_img).float(), torch.from_numpy(y_img).float()


class DataModuleDiffusion(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    # perform on every GPU
    def setup(self, stage: Optional[str] = None) -> None:
        X = sorted(list(DATA_ROOT.glob("**/*.npz")))

        # train_transforms = get_train_img_transforms()
        # val_transforms = get_val_img_transforms()
        preprocess = get_diffusion_preprocess()

        # self.train_dataset = DiffusionDataset(X_path=X[:-1] * 200, y_path=y[:-1] * 200, transform=preprocess)
        # self.val_dataset = DiffusionDataset(X_path=[X[-1]] * 4, y_path=[y[-1]] * 4, transform=preprocess)

        self.train_dataset = DiffusionDataset(path=X[:-1] * 200, X_transform=preprocess)
        self.val_dataset = DiffusionDataset(path=X[-1] * 4, X_transform=preprocess)  # *4 in order to allocate on 4 GPUs

    def train_dataloader(self):
        print(f"get {len(self.train_dataset)} training 3D image!")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        print(f"get {len(self.val_dataset)} validation 3D image!")
        return DataLoader(self.val_dataset, batch_size=1, num_workers=8)
