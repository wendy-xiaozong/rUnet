from typing import List
from monai.transforms import NormalizeIntensity, Compose, ToTensor, Resize, SpatialPad
from monai.transforms.compose import Transform
from utils.cropping import crop_to_nonzero
from utils.const import IMAGESIZE

import numpy as np


class Crop(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return crop_to_nonzero(img)


class Unsqueeze(Transform):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.expand_dims(img, axis=0)


class Squeeze(Transform):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.squeeze(img)


class Transpose(Transform):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        print(f"img shape: {img.shape}")
        return img.transpose([3, 0, 1, 2])


def get_diffusion_preprocess() -> List:
    return Compose(
        [
            NormalizeIntensity(nonzero=True),
            # Transpose(),
            # Resize((IMAGESIZE, IMAGESIZE, IMAGESIZE))
        ]
    )


def get_preprocess(is_label: bool) -> List:
    if not is_label:
        return [
            Crop(),
            # Use this instead of ScaleIntensity because of nnUnet.
            # But I don't think this should make a lot difference
            NormalizeIntensity(nonzero=True),
            # I really donno why I need to the unsqueeze things
            # maybe the way I use the data augmentation is the standard way(?)
            # but it works ¯\_(ツ)_/¯
            Unsqueeze(),
            SpatialPad(spatial_size=[193, 193, 193], method="symmetric", mode="constant"),
            Resize((IMAGESIZE, IMAGESIZE, IMAGESIZE)),
        ]
    else:
        return [
            Crop(),
            NormalizeIntensity(nonzero=True),
            Unsqueeze(),
            SpatialPad(spatial_size=[193, 193, 193], method="symmetric", mode="constant"),
            Resize((IMAGESIZE, IMAGESIZE, IMAGESIZE)),
        ]


def get_train_img_transforms() -> Compose:
    preprocess = get_preprocess(is_label=False)
    train_augmentation = [ToTensor()]
    return Compose(preprocess + train_augmentation)


def get_val_img_transforms() -> Compose:
    preprocess = get_preprocess(is_label=False)
    return Compose(preprocess + [ToTensor()])


def get_label_transforms() -> Compose:
    preprocess = get_preprocess(is_label=True)
    return Compose(preprocess + [ToTensor()])
