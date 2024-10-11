"""
This module contains a VGGFeatureExtractor class
for extracting features from the VGG19 network.
"""
import torch.nn as nn
from torchvision import models
from torchvision.models import vgg19, VGG19_Weights

class VGGFeatureExtractor(nn.Module):
    """
    A feature extractor using the VGG19 network pretrained on ImageNet,
    capable of computing content and style losses.
    """
    def __init__(self, content_layers, style_layers):
        super(VGGFeatureExtractor, self).__init__()
        # Load the pretrained VGG19 model
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.selected_layers = list(set(content_layers + style_layers))
        self.layers = nn.ModuleList()
        self.layer_names = []
        # Extract the specified layers
        for i, layer in enumerate(vgg19_model):
            self.layers.append(layer)
            if str(i) in self.selected_layers:
                self.layer_names.append(str(i))
            # Freeze the parameters
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the network to extract features from selected layers.
        Returns:
            content_features (dict): Features from content layers.
            style_features (dict): Features from style layers.
        """
        content_features = {}
        style_features = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            layer_index = str(i)
            if layer_index in self.content_layers:
                content_features[layer_index] = x
            if layer_index in self.style_layers:
                style_features[layer_index] = x
        return content_features, style_features
