# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def hard_sigmoid(x):
    return torch.nn.functional.relu6(x + 3) / 6


class DownsampledRegressionUnet(nn.Module):
    def __init__(
        self,
        encoder_name,
        encoder_weights,
        in_channels,
        unet_heads=1,
        num_output_bands=1,
        activation="hard_sigmoid",
        **kwargs,
    ):
        super().__init__()
        self.activation = activation
        self.num_output_bands = num_output_bands
        self.unet_heads = unet_heads

        # Remove unsupported kwargs
        if "classes" in kwargs:
            print("Ignoring classes argument.")
            kwargs.pop("classes")
        if "activation" in kwargs:
            print("Ignoring activation argument.")
            kwargs.pop("activation")

        # Filter out None values from kwargs
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=unet_heads,
            activation=None,
            **kwargs,
        )

        # Use average pooling for 8x downsampling
        self.downsample = nn.AvgPool2d(kernel_size=8, stride=8)

        self.final_conv = nn.Conv2d(
            unet_heads,
            num_output_bands,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        x = self.unet(x)
        x = self.downsample(x)
        x = self.final_conv(x)

        # Apply hard_sigmoid activation
        if self.activation == "hard_sigmoid":
            x = hard_sigmoid(x)

        return x
