"""
This module defines a ResNet-based backbone discriminator for oil painting style transfer.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetDiscriminator(nn.Module):
    """
    A ResNet-based backbone discriminator for style transfer using pretrained ResNet features.
    """
    def __init__(self, freeze_features=True):
        super(ResNetDiscriminator, self).__init__()

        # Load ResNet50 and use its feature extractor part
        resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Extract layers from ResNet up to the final convolutional block (before avgpool)
        self.features = nn.Sequential(*list(resnet_model.children())[:-4])

        # Freeze the feature extractor if specified
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False

        # Add a classifier for the final prediction
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """
        Forward pass through the ResNet backbone discriminator.
        """
        # Extract features using ResNet backbone
        x = self.features(x)

        # Pass the features through the classifier to get a real/fake score map
        x = self.classifier(x)

        return torch.sigmoid(x)
