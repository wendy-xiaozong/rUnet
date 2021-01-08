from typing import List
from monai.transforms import (
    NormalizeIntensity,
    # Resize,
    Compose,
    ToTensor,
)


def get_preprocess() -> List:
    return [
        # Use this instead of ScaleIntensity because of nnUnet.
        # But I don't think this should make a lot difference
        NormalizeIntensity(nonzero=True),
        # Resize((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)),
    ]


def get_train_transforms() -> Compose:
    preprocess = get_preprocess()
    train_augmentation = [
        ToTensor(),
    ]
    return Compose(preprocess + train_augmentation)


def get_val_transforms() -> Compose:
    preprocess = get_preprocess()
    return Compose(preprocess + [ToTensor()])
