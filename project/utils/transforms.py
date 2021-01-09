from typing import List
from monai.transforms import (
    NormalizeIntensity,
    # Resize,
    Compose,
    ToTensor,
)
from monai.transforms.compose import Transform
from utils.cropping import crop_to_nonzero

import numpy as np


class Crop(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return crop_to_nonzero(img)


def get_preprocess(is_train: bool) -> List:
    if is_train:
        return [
            Crop(),
            # Use this instead of ScaleIntensity because of nnUnet.
            # But I don't think this should make a lot difference
            NormalizeIntensity(nonzero=True),
            # Resize((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)),
        ]
    else:
        return [Crop()]


def get_train_transforms() -> Compose:
    preprocess = get_preprocess(is_train=True)
    train_augmentation = [
        ToTensor(),
    ]
    return Compose(preprocess + train_augmentation)


def get_val_transforms() -> Compose:
    preprocess = get_preprocess(is_train=False)
    return Compose(preprocess + [ToTensor()])
