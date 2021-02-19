from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from utils.const import DATA_ROOT

PATH = DATA_ROOT / "pred_true"


def plot_slices(
    processed_preds: List[ndarray], processed_targets: List[ndarray], clip_min: float, clip_max: float
) -> Any:
    fig: plt.Figure
    axes: plt.Axes
    fig, axes = plt.subplots(nrows=len(processed_preds), ncols=2)
    maes, maes_255, masked_maes, masked_255_maes = [], [], [], []
    max_value = 0
    min_value = 255
    for i, (pred, targ) in enumerate(zip(processed_preds, processed_targets)):
        mask = targ == 0
        pred_255 = np.clip(pred, -clip_min, clip_max)
        targ_255 = np.clip(targ, -clip_min, clip_max)
        min_pred = min(-clip_min, np.min(pred))
        min_targ = min(-clip_min, np.min(targ))
        pred_255 = np.floor(255 * ((pred_255 - min_pred) / (clip_max - min_pred)))
        targ_255 = np.floor(255 * ((targ_255 - min_targ) / (clip_max - min_targ)))
        max_value = max(max(max_value, np.max(pred_255)), np.max(targ_255))
        min_value = min(min(min_value, np.min(pred_255)), np.min(targ_255))

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

        axes[i][0].imshow(pred_255, cmap="Greys")
        axes[i][0].set_title(
            f"Predicted (MAE={mae_str}, MAE_255={mae_255_str}\n masked: (MAE={mask_str}, MAE_255={mask_255_str})"
            f"clip_max: {clip_max}, clip_min: {clip_min}",
            {"fontsize": 6},
        )
        axes[i][1].imshow(targ_255, cmap="Greys")
        axes[i][1].set_title("Target", {"fontsize": 8})
    mae_clean = "{:1.2f}".format(float(np.round(np.mean(maes), 2)))
    mae_255_clean = "{:1.2f}".format(float(np.round(np.mean(maes_255), 2)))
    mask_clean = "{:1.2f}".format(float(np.round(np.mean(masked_maes), 2)))
    mask_255_clean = "{:1.2f}".format(float(np.round(np.mean(masked_255_maes), 2)))
    fig.set_size_inches(w=8, h=12)
    fig.suptitle(
        f"All brains rescaled to [0, 255].\nAverage: (MAE_255={mae_255_clean}, MAE={mae_clean})\n Masked: (MAE_255={mask_255_clean}, MAE={mask_clean})"
    )
    fig.savefig(f"./plot/clip_max: {clip_max} clip_min:{clip_min}.png")
    print(f"max: {max_value}, min: {min_value}")
    return np.mean(masked_255_maes)


if __name__ == "__main__":
    brains = sorted(list(PATH.glob("*.npz")))

    targ_slices = []
    pred_slices = []
    for brain in brains:
        data = np.load(brain)
        target = data["target"]
        predict = data["predict"]
        cur_targ_slices = [target[64, ...], target[:, 64, :], target[..., 64]]
        cur_pred_slices = [predict[64, ...], predict[:, 64, :], predict[..., 64]]
        targs = np.concatenate(cur_targ_slices, axis=1)
        preds = np.concatenate(cur_pred_slices, axis=1)
        targ_slices.append(targs)
        pred_slices.append(preds)

    clips = [5, 4.5, 4, 3.5, 3, 2, 1.8]
    mae_dict = {}

    for clip_min in clips:
        for clip_max in clips:
            print(f"clip_min: {clip_min}, clip_max: {clip_max}")
            mae = plot_slices(pred_slices, targ_slices, clip_min, clip_max)
            mae_dict[f"clip_min:{clip_min}, clip_max:{clip_max}"] = mae

    print(mae_dict)

    for key, value in mae_dict.items():
        print(f"{key}: {value}")
