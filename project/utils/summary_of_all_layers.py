import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary

# from monai.networks.nets import UNet
from model.unet.unet import UNet
import torch


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.zeros(1, 1, 170, 200, 150)

        self.out_classes = 139
        self.deepth = 3
        self.kernal_size = 5  # whether this affect the model to learn?
        self.module_type = "Unet"
        self.downsampling_type = "max"
        self.normalization = "Batch"
        self.model = UNet(
            in_channels=1,
            num_encoding_blocks=self.hparams.deepth,
            out_channels_first_layer=self.hparams.out_channels_first_layer,
            kernal_size=self.hparams.kernel_size,
            normalization=self.hparams.normalization,
            module_type=self.hparams.model,
            downsampling_type=self.hparams.downsampling_type,
            dropout=0,
        )

    def forward(self, x):
        # return self.unet(x)
        return self.myModel(x)


class HighResNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.zeros(1, 1, 96, 96, 96)

        self.unet = HighResNet(in_channels=1, out_channels=139, dimensions=3)

    def forward(self, x):
        return self.unet(x)


if __name__ == "__main__":
    # HighResNet = HighResNetModel()
    # print("highResNet Model:")
    # print(ModelSummary(HighResNet, mode="full"))

    newNet = NewModel()
    print("newNet Model:")
    print(ModelSummary(newNet, mode="full"))
