import random
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random
import torch
import torch.nn.functional as F
from torch.nn import Sigmoid, MSELoss
from monai.losses import DiceLoss
from model.unet.unet import UNet
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils.visualize import log_all_info


class LitModel(pl.LightningModule):
    def __init__(self, hparams: AttributeDict):
        super(LitModel, self).__init__()
        self.hparams = hparams
        self.model = UNet(
            in_channels=1,
            out_classes=1,
            dimensions=3,
            padding_mode="zeros",
            activation="ReLU",
            conv_num_in_layer=[1, 2, 3, 3, 3],
            residual=False,
            out_channels_first_layer=16,
            kernal_size=5,
            normalization="Batch",
            downsampling_type="max",
        )
        # self.sigmoid = Sigmoid()
        self.criterion = MSELoss()
        # randomly pick one image to log
        self.train_log_step = random.randint(1, 500)
        self.val_log_step = random.randint(1, 100)

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1)) / np.prod(inputs.shape)
        if batch_idx == self.train_log_step:
            log_all_info(module=self, img=inputs[0], target=targets[0], preb=logits[0], loss=loss, batch_idx=batch_idx)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1)) / np.prod(inputs.shape)
        if batch_idx == self.train_log_step:
            log_all_info(module=self, img=inputs[0], target=targets[0], preb=logits[0], loss=loss, batch_idx=batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def validation_epoch_end(self, validation_step_outputs):
        self.train_log_step = random.randint(1, 500)
        self.val_log_step = random.randint(1, 100)

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
        # parser.add_argument("--loss", type=str, default="BCEWL", help="Loss Function")
        # parser.add_argument("--down_sample", type=str, default="max", help="the way to down sample")
        # parser.add_argument("--out_channels_first_layer", type=int, default=32, help="the first layer's out channels")
        # parser.add_argument("--deepth", type=int, default=4, help="the deepth of the unet")
        # parser.add_argument("--normalization", type=str, default="Batch")
        # parser.add_argument("--kernel_size", type=int, default=3, help="the kernal size")
        # parser.add_argument("--model", type=str, default="ResUnet")
        # parser.add_argument("--downsampling_type", type=str, default="max")
        return parser
