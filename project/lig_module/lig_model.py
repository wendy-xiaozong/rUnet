import random
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Sigmoid, BCEWithLogitsLoss
from monai.losses import DiceLoss
from model.unet.unet import VNet
from pytorch_lightning.metrics.functional.classification import dice_score
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils.visualize import log_all_info


class LitModel(pl.LightningModule):
    def __init__(self, hparams: AttributeDict):
        super(LitModel, self).__init__()
        self.hparams = hparams
        self.model = VNet(
            in_channels=1,
            # num_encoding_blocks=self.hparams.deepth,
            # out_channels_first_layer=self.hparams.out_channels_first_layer,
            out_channels_first_layer=16,
            # kernal_size=self.hparams.kernel_size,
            kernal_size=5,
            normalization=self.hparams.normalization,
            module_type=self.hparams.model,
            downsampling_type=self.hparams.downsampling_type,
        )
        self.sigmoid = Sigmoid()

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        inputs, targets = batch

        print(f"inputs shape:{inputs.shape}")
        print(f"targets shape:{targets.shape}")

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch

        print(f"inputs shape:{inputs.shape}")
        print(f"targets shape:{targets.shape}")

    def configure_optimizers(
        self,
    ) -> Tuple[List[Any], List[Dict]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        lr_dict = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=300, eta_min=0.000001),
            "monitor": "val_checkpoint_on",  # Default: val_loss
            "reduce_on_plateau": True,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_dict]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--loss", type=str, default="BCEWL", help="Loss Function")
        parser.add_argument("--down_sample", type=str, default="max", help="the way to down sample")
        parser.add_argument("--out_channels_first_layer", type=int, default=32, help="the first layer's out channels")
        parser.add_argument("--deepth", type=int, default=4, help="the deepth of the unet")
        parser.add_argument("--normalization", type=str, default="Batch")
        parser.add_argument("--kernel_size", type=int, default=3, help="the kernal size")
        parser.add_argument("--model", type=str, default="ResUnet")
        parser.add_argument("--downsampling_type", type=str, default="max")
        return parser
