from typing import Optional
import torch.nn as nn
from .conv import ConvolutionalBlock
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        num_encoding_blocks: int,
        residual: bool,
        kernal_size: int,
        normalization: str,
        downsampling_type: str,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
    ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        # self.dilation = initial_dilation
        is_first_block = True
        for i in range(num_encoding_blocks):  # 3
            encoding_block = EncodingBlock(
                in_channels,
                out_channels_first,
                out_channels=in_channels * 2,
                dimensions=dimensions,
                residual=residual,
                normalization=normalization,
                kernal_size=kernal_size,
                downsampling_type=downsampling_type,
                padding_mode=padding_mode,
                is_first_block=is_first_block,
                activation=activation,
                num_block=i,
            )
            is_first_block = False
            self.encoding_blocks.append(encoding_block)
            if dimensions == 2:
                in_channels = out_channels_first
                out_channels_first = in_channels * 2
            elif dimensions == 3:
                in_channels = out_channels_first
                out_channels_first = in_channels * 2

            self.out_channels = self.encoding_blocks[-1].out_channels

    def forward(self, x):
        skip_connections = []
        for encoding_block in self.encoding_blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        out_channels: int,
        dimensions: int,
        residual: bool,
        normalization: Optional[str],
        kernal_size: int = 5,
        downsampling_type: Optional[str] = "conv",
        is_first_block: bool = False,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        num_block: int = 0,
    ):
        super().__init__()

        self.num_block = num_block
        self.residual = residual

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels_first,
            normalization=normalization,
            kernal_size=kernal_size,
            padding_mode=padding_mode,
            activation=activation,
        )
        # print(f"conv1: in_channels:{in_channels}, out_channels_first:{out_channels_first}")
        out_channels_second = out_channels_first
        self.conv2 = ConvolutionalBlock(
            dimensions,
            out_channels_first,
            out_channels,
            normalization=normalization,
            kernal_size=kernal_size,
            padding_mode=padding_mode,
            activation=activation,
        )
        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions=dimensions,
                in_channels=in_channels,
                out_channels=out_channels_second,
                kernal_size=1,
                normalization=None,
                activation=None,
            )

        self.downsampling_type = downsampling_type

        self.downsample = None
        if downsampling_type == "max":
            self.downsample = get_downsampling_maxpooling_layer(dimensions, downsampling_type)
        elif downsampling_type == "conv":
            self.downsample = get_downsampling_conv_layer(in_channels=out_channels_second)

        # self.out_channels = self.conv2.conv_layer.out_channels
        self.out_channels = out_channels_second

    def forward(self, x):
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
        return x, skip_connection


def get_downsampling_maxpooling_layer(
    dimensions: int,
    pooling_type: str,
    kernel_size: int = 2,
    stride: int = 2,
):
    class_name = "{}Pool{}d".format(pooling_type.capitalize(), dimensions)
    class_ = getattr(nn, class_name)
    return class_(kernel_size)


def get_downsampling_conv_layer(
    in_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
):
    class_name = "Conv3d"
    class_ = getattr(nn, class_name)
    return class_(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride)
