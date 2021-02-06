from argparse import ArgumentParser
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sigmoid, MSELoss, Softmax
from monai.losses import DiceLoss
from model.unet.unet import UNet
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.visualize import log_all_info


def scale_img_to_0_255(img: np.ndarray, imin: Any = None, imax: Any = None) -> np.ndarray:
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) * (1 / (imax - imin))) * 255, dtype="uint8")
    return scaled


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
            use_sigmoid=True,
        )
        self.sigmoid = Sigmoid()
        self.criterion = MSELoss()
        # randomly pick one image to log
        self.train_log_step = random.randint(1, 500)
        self.val_log_step = random.randint(1, 100)

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x / (1 - x))

    def training_step(self, batch, batch_idx: int):
        inputs, targets = batch

        logits = self(inputs)
        targets = self.sigmoid(targets)
        loss = self.criterion(logits.view(-1), targets.view(-1)) / np.prod(inputs.shape)
        if batch_idx == self.train_log_step:
            log_all_info(
                module=self,
                img=self.logit(inputs[0]),
                target=self.logit(targets[0]),
                preb=self.logit(logits[0]),
                loss=loss,
                batch_idx=batch_idx,
                state="train",
            )
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch

        logits = self(inputs)
        targets = self.sigmoid(targets)
        loss = self.criterion(logits.view(-1), targets.view(-1)) / np.prod(inputs.shape)
        if batch_idx == self.train_log_step:
            log_all_info(
                module=self,
                img=self.logit(inputs[0]),
                target=self.logit(targets[0]),
                preb=self.logit(logits[0]),
                loss=loss,
                batch_idx=batch_idx,
                state="val",
            )
        self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)

    def validation_epoch_end(self, validation_step_outputs):
        self.train_log_step = random.randint(1, 500)
        self.val_log_step = random.randint(1, 100)

    def test_step(self, batch, batch_idx: int):
        inputs, targets = batch
        logits = self(inputs)

        inputs = scale_img_to_0_255(inputs.cpu().detach().numpy().squeeze())
        num_non_zero = np.count_nonzero(inputs)
        targets = scale_img_to_0_255(targets.cpu().detach().numpy().squeeze())
        predicts = scale_img_to_0_255(logits.cpu().detach().numpy().squeeze())
        predicts -= predicts[0][0][0]
        brain_mask = inputs == inputs[0][0][0]
        predicts[brain_mask] = 0
        if batch_idx == 1:
            log_all_info(
                module=self, img=inputs, target=targets, preb=predicts, loss=0.0, batch_idx=batch_idx, state="tmp"
            )
        diff_tensor = np.absolute(predicts - targets)
        diff_average = np.sum(diff_tensor) / num_non_zero
        return {"diff_average": diff_average}

    def test_epoch_end(self, test_step_outputs):
        average = np.mean(test_step_outputs[0]["diff_average"])
        print(f"average absolute error: {average}")
        return average

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        # scheduler = ReduceLROnPlateau(optimizer, threshold=1e-10)
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
