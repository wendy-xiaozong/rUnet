from typing import List

import numpy as np
from numpy import ndarray
from matplotlib.pyplot import plot as plt
from utils.const import DATA_ROOT

PATH = DATA_ROOT / "pred_true"


def plot_slices(processed_preds: List[ndarray], processed_targets: List[ndarray], mask_pred: bool = True) -> None:
    fig: plt.Figure
    axes: plt.Axes
    fig, axes = plt.subplots(nrows=len(processed_preds), ncols=2)
    maes, maes_255, masked_maes, masked_255_maes = [], [], [], []
    for i, (pred, targ) in enumerate(zip(processed_preds, processed_targets)):
        mask = targ == 0
        CLIP_MIN, CLIP_MAX = 4, 5
        pred_255 = np.clip(pred, -CLIP_MIN, CLIP_MAX) + CLIP_MIN
        targ_255 = np.clip(targ, -CLIP_MIN, CLIP_MAX) + CLIP_MIN
        # pred_255[mask] = 0
        # targ_255[mask] = 0
        pred_255 = np.floor(256 * ((pred - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)))
        targ_255 = np.floor(256 * ((targ - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)))
        pred_255[mask] = 0
        targ_255[mask] = 0
        mae_255 = np.mean(np.abs(pred_255.ravel() - targ_255.ravel()))
        mae = np.mean(np.abs(pred.ravel() - targ.ravel()))
        masked_mae = np.mean(np.abs(pred[~mask].ravel() - targ[~mask].ravel()))
        masked_255_mae = np.mean(np.abs(pred_255[~mask].ravel() - targ_255[~mask].ravel()))
        maes.append(mae)
        maes_255.append(mae_255)
        masked_maes.append(masked_mae)
        masked_255_maes.append(masked_255_mae)
        mae_str = "{:1.2f}".format(float(np.round(mae, 2)))
        mae_255_str = "{:1.2f}".format(float(np.round(mae_255, 2)))
        mask_str = "{:1.2f}".format(float(np.round(masked_mae, 2)))
        mask_255_str = "{:1.2f}".format(float(np.round(masked_255_mae, 2)))
        axes[i][0].imshow(pred_255, cmap="Greys", vmax=0)
        axes[i][0].set_title(
            f"Predicted (MAE={mae_str}, MAE_255={mae_255_str}\n masked: (MAE={mask_str}, MAE_255={mask_255_str})",
            {"fontsize": 8},
        )
        axes[i][1].imshow(targ_255, cmap="Greys", vmax=0)
        axes[i][1].set_title("Target", {"fontsize": 8})
    mae_clean = "{:1.2f}".format(float(np.round(np.mean(maes), 2)))
    mae_255_clean = "{:1.2f}".format(float(np.round(np.mean(maes_255), 2)))
    mask_clean = "{:1.2f}".format(float(np.round(np.mean(masked_maes), 2)))
    mask_255_clean = "{:1.2f}".format(float(np.round(np.mean(masked_255_maes), 2)))
    fig.set_size_inches(w=8, h=12)
    fig.suptitle(
        f"All brains rescaled to [0, 255].\nAverage: (MAE_255={mae_255_clean}, MAE={mae_clean})\n Masked: (MAE_255={mask_255_clean}, MAE={mask_clean})"
    )
    plt.show()


if __name__ == "__main__":
    brains = sorted(list(PATH.glob("*.npz")))
