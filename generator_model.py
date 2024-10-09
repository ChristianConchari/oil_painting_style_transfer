import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A convolutional block used in the generator model for oil painting style transfer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downsample (bool): Whether to use a downsampling convolution or upsampling transpose convolution. Default is True.
        use_activation (bool): Whether to use ReLU activation. Default is True.
        **kwargs: Additional arguments for the convolutional layers.

    Attributes:
        conv (nn.Sequential): A sequential container of layers including convolution or transpose convolution,
                              instance normalization, and ReLU activation or identity.
    """
    def __init__(self, in_channels, out_channels, downsample=True, use_activation=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if downsample
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if use_activation else nn.Identity()
        )

    def forward(self, x):
        """
        Defines the forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the convolutional layers.
        """
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    """
    A residual block used in the generator model for oil painting style transfer.

    Args:
        channels (int): Number of input and output channels.

    Attributes:
        block (nn.Sequential): A sequential container of two convolutional blocks used to create a residual connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_activation=False, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after adding the residual connection.
        """
        x = x + self.block(x)
        return x
    
class Generator(nn.Module):
    """
    This class defines a generator neural network model for oil painting style transfer.

    Args:
        img_channels (int): Number of input and output image channels.
        num_features (int): Number of features for the convolutional layers. Default is 64.
        num_residuals (int): Number of residual blocks in the model. Default is 9.

    Attributes:
        initial (nn.Sequential): The initial convolutional layer followed by ReLU activation.
        downsample_block (nn.ModuleList): A list of downsampling convolutional blocks.
        residual_blocks (nn.Sequential): A series of residual blocks.
        upsample_block (nn.ModuleList): A list of upsampling convolutional blocks.
        last (nn.Conv2d): The final convolutional layer to produce the output image.
    """
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        
        self.downsample_block = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        
        self.upsample_block = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, downsample=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features, downsample=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )
        
        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        
    def forward(self, x):
        """
        Defines the forward pass of the generator model.

        Args:
            x (torch.Tensor): Input tensor representing the input image.

        Returns:
            torch.Tensor: Output tensor representing the generated image after passing through all layers.
        """
        x = self.initial(x)
        
        for layer in self.downsample_block:
            x = layer(x)
            
        x = self.residual_blocks(x)
        
        for layer in self.upsample_block:
            x = layer(x)
            
        x = torch.tanh(self.last(x))
        
        return x
  