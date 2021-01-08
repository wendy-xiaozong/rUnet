"""Some code is borrowed and adapted from:
https://github.com/DM-Berger/unet-learn/blob/6dc108a9a6f49c6d6a50cd29d30eac4f7275582e/src/lightning/log.py
https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
"""
import sys
import os
import torch

from numpy import ndarray
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union, Optional
from pytorch_lightning.core.lightning import LightningModule
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.pyplot import Axes, Figure
from matplotlib.text import Text
from numpy import ndarray
from tqdm.auto import tqdm, trange

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

"""
For TensorBoard logging usage, see:
https://www.tensorflow.org/api_docs/python/tf/summary
For Lightning documentation / examples, see:
https://pytorch-lightning.readthedocs.io/en/latest/experiment_logging.html#tensorboard
NOTE: The Lightning documentation here is not obvious to newcomers. However,
`self.logger` returns the Torch TensorBoardLogger object (generally quite
useless) and `self.logger.experiment` returns the actual TensorFlow
SummaryWriter object (e.g. with all the methods you actually care about)
For the Lightning methods to access the TensorBoard .summary() features, see
https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.html#pytorch_lightning.loggers.TensorBoardLogger
**kwargs for SummaryWriter constructor defined at
https://www.tensorflow.org/api_docs/python/tf/summary/create_file_writer
^^ these args look largely like things we don't care about ^^
"""


def make_imgs(img: ndarray, imin: Any = None, imax: Any = None) -> ndarray:
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) / (imax - imin)) * 255, dtype=np.uint8)
    return scaled


# what this function doing?
def turn(array_2d: np.ndarray) -> np.ndarray:
    return np.flipud(np.rot90(array_2d))


# https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data
class Slices:
    def __init__(self, lightning: LightningModule, input: Tensor, target: Tensor, pred: Tensor):
        self.lightning = lightning
        self.input_img = input.cpu().detach().numpy()  # shape: (10, 3, 256, 256)
        self.gt_img = target.cpu().detach().numpy()
        self.pred_img = pred.cpu().detach().numpy()

        self.slices = [self.input_img, self.gt_img, self.pred_img]

    def plot(self) -> Figure:
        nrows, ncols = 3, self.input_img.shape[0]
        plot_width, plot_length = 125, 50

        fig = plt.figure(figsize=(plot_width, plot_length))
        gs = gridspec.GridSpec(nrows, ncols)

        for idx in range(nrows):
            axes = [plt.subplot(gs[idx * ncols + i]) for i in range(ncols)]
            self.plot_row(self.slices[idx], axes, idx)

        plt.tight_layout()
        return fig

    def plot_row(
        self,
        slices: List,
        axes: Tuple[Any, Any, Any],
        row_num: int,
    ) -> None:
        for idx, (slice, axis) in enumerate(zip(slices, axes)):
            img = make_imgs(slice)
            if row_num == 0:
                img = np.transpose(img, (1, 2, 0))
                axis.imshow(img)
            else:
                axis.imshow(img, cmap="gray")
            axis.grid(False)
            axis.set_xticks([])
            axis.set_yticks([])

    def log(self, fig: Figure, dice_score: float) -> None:
        logger = self.lightning.logger
        summary = f"fold:{self.lightning.hparams.fold}-run:{self.lightning.hparams.run}-epoch:{self.lightning.current_epoch + 1}-dice_score:{dice_score:0.5f}"
        logger.experiment.add_figure(summary, fig, close=True)


def log_all_info(module: LightningModule, img: Tensor, target: Tensor, pred: Tensor, dice_score: float) -> None:
    """Helper for decluttering training loop. Just performs all logging functions."""
    slice = Slices(module, img, target, pred)
    fig = slice.plot()

    slice.log(fig, dice_score)
