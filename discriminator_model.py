"""
This module defines the discriminator model for oil painting style transfer.
"""
import torch
from torch import nn
from blocks import ConvBlock

class Discriminator(nn.Module):
    """
    Discriminator model for oil painting style transfer.
    This class defines a discriminator neural network model using convolutional layers.
    The discriminator is used to distinguish between real and generated images.
    
    Attributes:
        initial (nn.Sequential): Initial convolutional layer followed by a LeakyReLU activation.
        model (nn.Sequential): Sequential container of convolutional blocks.
        output (nn.Conv2d): Final convolutional layer that outputs a single-channel tensor.
        
    Methods:
        __init__(in_channels=3, alpha=0.2, features=None):
            Initializes the Discriminator model with the given parameters.
        forward(x):
            Defines the forward pass of the discriminator model.
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
        
        # Assign the model as a sequential container of the layers
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
    