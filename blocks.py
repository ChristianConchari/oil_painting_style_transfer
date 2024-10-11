import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A convolutional block used in the discriminator model for oil painting style transfer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride size for the convolution.

    Attributes:
        conv (nn.Sequential): A sequential container of layers including spectral normalization,
                              convolution, batch normalization, and LeakyReLU activation.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        upsample=False,
        use_activation=True,
        leaky_relu=False,
        reflection_pad=True,
        **kwargs
    ):
        super().__init__()
        # Convolutional layer
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding_mode='reflect' if reflection_pad else 'zeros',
                **kwargs
            ) if not upsample
            else nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                **kwargs
            ),
            nn.BatchNorm2d(out_channels),
        )
        # Use LeakyReLU activation if specified
        self.activation = nn.Identity() if not use_activation else nn.LeakyReLU(0.2) if leaky_relu else nn.ReLU()

    def forward(self, x, skip_block_output=None):
        """
        Defines the forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the convolutional layers.
        """
        x = self.conv_block(x)
        if skip_block_output is not None:
            x += skip_block_output
        return self.activation(x)
    
class ResidualBlock(nn.Module):
    """
    A residual block used in the generator model for oil painting style transfer.

    Args:
        channels (int): Number of input and output channels.

    Attributes:
        block (nn.Sequential): A sequential container of two convolutional blocks used to create a residual connection.
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        # Define two convolutional blocks to create the residual connection
        self.block = nn.Sequential(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                reflection_pad=False,
            ),
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_activation=False,
                reflection_pad=False,
            ),
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after adding the residual connection.
        """
        return x + self.block(x)
