"""
This module defines the discriminator model for oil painting style transfer.
The discriminator model is a convolutional neural network that takes an image
as input and outputs a single scalar value. The model is used to distinguish
between real and generated images. The model is defined using spectral normalization
and LeakyReLU activation functions.
"""
import torch
from torch import nn
from blocks import ConvBlock

class Discriminator(nn.Module):
    """
    This class defines a discriminator neural network model for oil painting style transfer.

    Args:
        in_channels (int): Number of input channels. Default is 3.
        features (list, optional): List of feature sizes for each layer. Default is [64, 128, 256, 512].

    Attributes:
        initial (nn.Sequential): The initial layer of the discriminator consisting 
            of a spectral normalized convolutional layer followed by a LeakyReLU 
            activation.
        model (nn.Sequential): The sequential container of the discriminator 
            containing a series of blocks and a final spectral normalized 
            convolutional layer.
    """
    def __init__(self, in_channels=3, alpha=0.2, features=None):
        # Default feature sizes for the discriminator
        if features is None:
            features = [64, 128, 256, 512]
        super().__init__()
        # Initial convolutional layer (no normalization)
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect',
            ),
            nn.LeakyReLU(alpha),
        )
        
        # List to store the subsequent layers
        layers = []
        # Set initial number of input channels
        in_channels = features[0]
        
        for feature in features[1:]:
            # Use stride of 1 for the last feature layer, otherwise use stride of 2
            layers.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=4,
                    stride=1 if feature == features[-1] else 2
                )
            )
            # Update the number of input channels for the next block
            in_channels = feature
        
        self.model = nn.Sequential(*layers)
        
        # Add the final convolutional layer
        self.output = nn.Conv2d(
            in_channels,
            1,
            kernel_size=4,
            stride=1,
            padding=1,
            padding_mode='reflect'
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the discriminator model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the initial and model layers.
        """
        x = self.initial(x)
        x = self.model(x)
        return torch.sigmoid(self.output(x))
    