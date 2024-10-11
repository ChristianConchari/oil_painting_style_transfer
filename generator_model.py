"""
This file contains the implementation of the Generator model used in the CycleGAN architecture.
"""
import torch
import torch.nn as nn
from blocks import ConvBlock, ResidualBlock
  
class Generator(nn.Module):
    """
    Generator Model for Image Style Transfer
    This class defines a generator model used for image style transfer, which includes
    encoder blocks for downsampling, residual blocks for feature enhancement, and 
    decoder blocks for upsampling the image back to its original size.
    
    Attributes:
        encoder_blocks (nn.ModuleList): A list of convolutional blocks used for downsampling the input image.
        bottleneck (nn.Sequential): A sequence of residual blocks used to enhance feature representation.
        decoder_blocks (nn.ModuleList): A list of convolutional blocks used for upsampling the image.
        output (nn.Conv2d): The final convolutional layer that produces the output image.
        
    Methods:
        forward(x): Defines the forward pass of the generator model
    """
    def __init__(self, img_channels, num_features=64, num_residuals=6):
        super().__init__()

        # Encoder block to downsample the image
        self.encoder_blocks = nn.ModuleList([
            ConvBlock(
                img_channels,
                num_features,
                kernel_size=7,
                padding=3,
                stride=1,
                use_activation=False,
                reflection_pad=False
            ),
            ConvBlock(
                num_features,
                num_features * 2,
                kernel_size=3,
                padding=1,
                stride=2,
                reflection_pad=False
            ),
            ConvBlock(
                num_features * 2,
                num_features * 4,
                kernel_size=3,
                padding=1,
                stride=2,
                reflection_pad=False
            )
        ])

        # Residual blocks to enhance feature representation
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        # Decoder blocks to upsample the image
        self.decoder_blocks = nn.ModuleList([
            ConvBlock(
                num_features * 4,
                num_features * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                upsample=True,
            ),
            ConvBlock(
                num_features * 4,
                num_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                upsample=True,
            ),
            ConvBlock(
                num_features * 2,
                num_features,
                kernel_size=4,
                stride=2,
                padding=1,
                upsample=True,
            )
        ])

        # Final layer to produce the output image
        self.output = nn.Conv2d(
            num_features,
            img_channels,
            kernel_size=7,
            padding=3,
            padding_mode='reflect'
        )

    def forward(self, x):
        """
        Defines the forward pass of the generator model.

        Args:
            x (torch.Tensor): Input tensor representing the input image.

        Returns:
            torch.Tensor: Output tensor representing the generated image after passing through all layers.
        """
        # Pass through downsampling layers
        for block in self.encoder_blocks:
            x = block(x)
        # Pass through residual blocks
        x = self.bottleneck(x)
        # Pass through upsampling layers
        for block in self.decoder_blocks:
            x = block(x)
        # Pass through the final layer with Tanh activation to get the output image
        return torch.tanh(self.output(x))
