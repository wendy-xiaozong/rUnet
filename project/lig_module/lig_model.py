from argparse import ArgumentParser
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random
import torch
import seaborn as sns
import matplotlib.pyplot as plt
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
    scaled[scaled < 0] = 0
    scaled[scaled >= 255] = 255
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
            use_sigmoid=False,
            use_bias=True,
        )
        self.sigmoid = Sigmoid()
        self.criterion = MSELoss()
        self.train_log_step = random.randint(1, 500)
        self.val_log_step = random.randint(1, 100)

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1)) / np.prod(inputs.shape)
        # if batch_idx == self.train_log_step:
        #     log_all_info(
        #         module=self,
        #         img=inputs[0],
        #         target=targets[0],
        #         preb=logits[0],
        #         loss=loss,
        #         batch_idx=batch_idx,
        #         state="train",
        #     )
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        # return {"loss": loss}
        return {}

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch

        logits = self(inputs)
        loss = self.criterion(logits.view(-1), targets.view(-1)) / np.prod(inputs.shape)
        self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)

        inputs = inputs.cpu().detach().numpy().squeeze()
        targets = targets.cpu().detach().numpy().squeeze()
        predicts = logits.cpu().detach().numpy().squeeze()

        # brain_mask = inputs == inputs[0][0][0]
        # predicts = predicts[~brain_mask]
        # targets = targets[~brain_mask]

        if batch_idx in [4, 6, 10, 12, 13]:
            fig, ax = plt.subplots(3, 1, figsize=(15, 25))
            sns.distplot(targets, kde=True, ax=ax[0])
            sns.distplot(predicts, kde=True, ax=ax[1])
            diff = predicts - targets
            sns.histplot(diff, kde=True, ax=ax[2])
            ax[0].set_title("targets")
            ax[1].set_title("predicts")
            ax[2].set_title("difference")
            fig.savefig(f"/home/jueqi/projects/def-jlevman/jueqi/rUnet/3/predicts_and_targets_{batch_idx}.png")
            np.savez(f"{batch_idx}.npz", target=targets, predict=predicts)

        # percents = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.7, 1, 2, 5]
        # MAEs = []
        # for percent_1 in percents:
        #     for percent_2 in percents:
        #         predicts = scale_img_to_0_255(
        #             predicts, imin=np.percentile(predicts, q=percent_1), imax=np.percentile(predicts, q=100 - percent_1)
        #         )
        #         targets = scale_img_to_0_255(
        #             targets, imin=np.percentile(targets, q=percent_2), imax=np.percentile(targets, q=100 - percent_2)
        #         )
        #         diff_tensor = np.absolute(predicts - targets)
        #         diff_average = np.mean(diff_tensor)
        #         MAEs.append(diff_average)

        # return_dict = {}
        # for id, MAE in enumerate(MAEs):
        #     return_dict[f"diff_average_{id}"] = MAE
        return {}

    def validation_epoch_end(self, validation_step_outputs):
        self.train_log_step = random.randint(1, 500)
        self.val_log_step = random.randint(1, 100)

        # for i in range(100):
        #     average = np.mean(validation_step_outputs[0][f"diff_average_{i}"])
        #     print(f"average absolute error for No. {i}: {average}")

    def test_step(self, batch, batch_idx: int):
        inputs, targets = batch
        logits = self(inputs)

        if batch_idx == 1:
            log_all_info(
                module=self,
                img=inputs[0],
                target=targets[0],
                preb=logits[0],
                loss=0.0,
                batch_idx=batch_idx,
                state="test",
            )

        inputs = inputs.cpu().detach().numpy().squeeze()
        targets = targets.cpu().detach().numpy().squeeze()
        predicts = logits.cpu().detach().numpy().squeeze()

        brain_mask = inputs == inputs[0][0][0]
        predicts = predicts[~brain_mask]
        targets = targets[~brain_mask]

        percents = [0.001, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.1]
        MAEs = []
        for percent_1 in percents:
            for percent_2 in percents:
                predicts = scale_img_to_0_255(
                    predicts, imin=np.percentile(predicts, q=percent_1), imax=np.percentile(predicts, q=100 - percent_1)
                )
                targets = scale_img_to_0_255(
                    targets, imin=np.percentile(targets, q=percent_2), imax=np.percentile(targets, q=100 - percent_2)
                )
                diff_tensor = np.absolute(predicts - targets)
                diff_average = np.mean(diff_tensor)
                MAEs.append(diff_average)
        return_dict = {}
        for id, MAE in enumerate(MAEs):
            return_dict[f"diff_average_{id}"] = MAE
        return return_dict

    def test_epoch_end(self, test_step_outputs):
        for i in range(100):
            average = np.mean(test_step_outputs[0][f"diff_average_{i}"])
            print(f"average absolute error for No. {i}: {average}")

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
        parser.add_argument("--learning_rate", type=float, default=1e-2)
        # parser.add_argument("--loss", type=str, default="BCEWL", help="Loss Function")
        # parser.add_argument("--down_sample", type=str, default="max", help="the way to down sample")
        # parser.add_argument("--out_channels_first_layer", type=int, default=32, help="the first layer's out channels")
        # parser.add_argument("--deepth", type=int, default=4, help="the deepth of the unet")
        # parser.add_argument("--normalization", type=str, default="Batch")
        # parser.add_argument("--kernel_size", type=int, default=3, help="the kernal size")
        # parser.add_argument("--model", type=str, default="ResUnet")
        # parser.add_argument("--downsampling_type", type=str, default="max")
        return parser
