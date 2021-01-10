# -*- coding: utf-8 -*-

"""Main module."""

from typing import Optional
import torch
import torch.nn as nn
from .encoding import Encoder, EncodingBlock
from .decoding import Decoder
from .conv import ConvolutionalBlock

# I might need to try 2d data


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_classes: int,
        dimensions: int,
        num_encoding_blocks: int,
        out_channels_first_layer: int,
        kernal_size: int,
        normalization: str,
        downsampling_type: str,
        residual: bool,
        padding_mode: str,
        activation: Optional[str],
        use_softmax: bool = False,
    ):
        super().__init__()
        depth = num_encoding_blocks
        self.use_softmax = use_softmax

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels_first=out_channels_first_layer,
            dimensions=dimensions,
            num_encoding_blocks=depth,
            residual=residual,
            kernal_size=kernal_size,
            normalization=normalization,
            downsampling_type=downsampling_type,
            padding_mode=padding_mode,
            activation=activation,
        )

        # There is only one layer at the bottom of the Unet
        in_channels = self.encoder.out_channels
        in_channels_skip_connection = in_channels
        out_channels_first = in_channels * 2
        self.bottom_block = EncodingBlock(
            in_channels=in_channels,
            out_channels_first=out_channels_first,
            out_channels=in_channels,
            dimensions=dimensions,
            residual=residual,
            normalization=normalization,
            kernal_size=kernal_size,
            padding_mode=padding_mode,
            activation=activation,
        )

        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type="conv",
            num_decoding_blocks=num_decoding_blocks,
            kernal_size=kernal_size,
            residual=residual,
            normalization=normalization,
            padding_mode=padding_mode,
            activation=activation,
        )

        in_channels = out_channels_first_layer
        self.classifier = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels=out_classes,
            kernal_size=1,
            activation=None,
            normalization=None,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections, encoding = self.encoder(x)
        # print(f"first skip connection shape: {skip_connections[0].shape}")
        x = self.bottom_block(encoding)
        x = self.decoder(skip_connections, x)
        x = self.classifier(x)
        if self.use_softmax:
            return self.softmax(x)
        else:
            return x


class UNet2D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs["dimensions"] = 2
        kwargs["num_encoding_blocks"] = 5
        kwargs["out_channels_first_layer"] = 64
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class UNet3D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs["dimensions"] = 3
        kwargs["num_encoding_blocks"] = 3  # 4
        kwargs["out_channels_first_layer"] = 8
        # kwargs['normalization'] = 'batch'
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class VNet(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs["dimensions"] = 3
        kwargs["num_encoding_blocks"] = 4  # 4
        kwargs["out_channels_first_layer"] = 16
        kwargs["kernal_size"] = 5
        kwargs["residual"] = True
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)
