import torch 
import torch.nn as nn
from blocks import ConvBlock, ResidualBlock
    
class Generator(nn.Module):
    """
    This class defines a generator neural network model for oil painting style transfer.

    Args:
        img_channels (int): Number of input and output image channels.
        num_features (int): Number of features for the convolutional layers. Default is 64.
        num_residuals (int): Number of residual blocks in the model. Default is 9.

    Attributes:
        initial (nn.Sequential): The initial convolutional layer followed by InstanceNorm2d and LeakyReLU activation.
        downsample_block (nn.ModuleList): A list of downsampling convolutional blocks.
        residual_blocks (nn.Sequential): A series of residual blocks.
        upsample_block (nn.ModuleList): A list of upsampling convolutional blocks.
        last (nn.Conv2d): The final convolutional layer to produce the output image.
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
