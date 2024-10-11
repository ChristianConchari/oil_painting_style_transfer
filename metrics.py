"""
This module contains functions for evaluating the performance of the model.
"""
from PIL import Image
import os
from torchvision import transforms
import torch
from torch import nn
from torchvision.utils import save_image
import config
from tqdm import tqdm
from torchmetrics.image.inception import InceptionScore
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import random
from torcheval.metrics import FrechetInceptionDistance
from datasets import ImageFolderDataset
from vgg_feature_extractor import VGGFeatureExtractor
from utils import gram_matrix

def compute_fid(real_images_path, generated_images_path, batch_size=50, dims=2048):
    """
    Computes the FID between real images and generated images using torcheval.
    """
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize to match InceptionV3 input size
        transforms.ToTensor(),
    ])

    # Load real and generated images
    real_dataset = ImageFolderDataset(real_images_path, transform)
    generated_dataset = ImageFolderDataset(generated_images_path, transform)

    # Create data loaders
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    generated_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the FID metric
    fid_metric = FrechetInceptionDistance(feature_dim=dims).to(config.DEVICE)

    # Update the metric with real images
    for real_batch in real_loader:
        if real_batch is None or len(real_batch) == 0:
            continue
        real_batch = real_batch.to(config.DEVICE)
        fid_metric.update(real_batch, is_real=True)

    # Update the metric with generated images
    for gen_batch in generated_loader:
        if gen_batch is None or len(gen_batch) == 0:
            continue
        gen_batch = gen_batch.to(config.DEVICE)
        fid_metric.update(gen_batch, is_real=False)

    # Compute the FID score
    fid_value = fid_metric.compute()
    print(f"FID score between {real_images_path} and {generated_images_path}: {fid_value.item()}")
    return fid_value.item()

def compute_inception_score(generated_images_path, batch_size=50):
    """
    Computes the Inception Score (IS) for generated images.
    """
    # Define the transformation consistent with previous code
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 input size
        transforms.ToTensor(),          # Converts to [0,1] range
    ])

    # Use the custom dataset class
    dataset = ImageFolderDataset(generated_images_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the InceptionScore metric
    inception_score_metric = InceptionScore().to(config.DEVICE)

    # Loop through the DataLoader and update the metric
    for batch in loader:
        if batch is None or len(batch) == 0:
            continue

        # Scale images to [0, 255]
        batch = batch * 255
        batch = batch.to(torch.uint8)
        batch = batch.to(config.DEVICE)

        inception_score_metric.update(batch)

    # Compute the Inception Score
    is_mean, is_std = inception_score_metric.compute()

    print(f"Inception Score for {generated_images_path}: {is_mean.item()} ± {is_std.item()}")
    return is_mean.item(), is_std.item()

def compute_content_loss_over_dataset(
    generated_images_path,
    content_images_path,
    batch_size=1,
    layer='21'
):
    """
    Computes the average content loss over the dataset.
    """
    device = config.DEVICE

    # Instantiate the feature extractor
    content_layers = [layer]
    feature_extractor = VGGFeatureExtractor(content_layers, []).to(device)
    feature_extractor.eval()

    # Define transformations consistent with VGG19 requirements
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Use the custom dataset class
    generated_dataset = ImageFolderDataset(generated_images_path, transform=transform)
    content_dataset = ImageFolderDataset(content_images_path, transform=transform)

    generated_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    content_loader = DataLoader(content_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Preprocessing normalization for VGG19
    vgg_preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    # Initialize variables
    total_content_loss = 0.0
    count = 0

    with torch.no_grad():
        for gen_batch, content_batch in zip(generated_loader, content_loader):
            if gen_batch is None or content_batch is None:
                continue
            gen_batch = gen_batch.to(device)
            content_batch = content_batch.to(device)

            # Apply VGG19 preprocessing
            gen_batch = vgg_preprocess(gen_batch)
            content_batch = vgg_preprocess(content_batch)

            # Extract features
            gen_content_feats, _ = feature_extractor(gen_batch)
            content_content_feats, _ = feature_extractor(content_batch)

            # Get features from the specified layer
            gen_features = gen_content_feats[layer]
            content_features = content_content_feats[layer]

            # Compute content loss
            loss = nn.functional.mse_loss(gen_features, content_features)
            total_content_loss += loss.item()
            count += 1

    average_content_loss = total_content_loss / count if count > 0 else 0
    print(f"Average Content Loss: {average_content_loss}")
    return average_content_loss

def compute_style_loss_over_dataset(
    generated_images_path,
    style_images_path,
    batch_size=1,
    layers=None
):
    """
    Computes the average style loss over the dataset.
    """
    if layers is None:
        layers = ['0', '5', '10', '19', '28']
        
    device = config.DEVICE

    # Instantiate the feature extractor
    content_layers = []  # No content layers needed for style loss
    style_layers = layers  # Use specified layers for style features
    feature_extractor = VGGFeatureExtractor(content_layers, style_layers).to(device)
    feature_extractor.eval()

    # Define transformations consistent with VGG19 requirements
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # No need to normalize here; we'll do it after batching
    ])

    # Use the custom dataset class
    generated_dataset = ImageFolderDataset(generated_images_path, transform=transform)
    style_dataset = ImageFolderDataset(style_images_path, transform=transform)

    generated_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    style_loader = DataLoader(style_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Preprocessing normalization for VGG19
    vgg_preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    # Initialize variables
    total_style_loss = 0.0
    count = 0

    with torch.no_grad():
        for gen_batch, style_batch in zip(generated_loader, style_loader):
            if gen_batch is None or style_batch is None:
                continue
            gen_batch = gen_batch.to(device)
            style_batch = style_batch.to(device)

            # Apply VGG19 preprocessing
            gen_batch = vgg_preprocess(gen_batch)
            style_batch = vgg_preprocess(style_batch)

            # Extract features
            _, gen_style_feats = feature_extractor(gen_batch)
            _, style_style_feats = feature_extractor(style_batch)

            # Compute style loss for the specified layers
            style_loss = 0.0
            for layer in layers:
                gen_feature = gen_style_feats[layer]
                style_feature = style_style_feats[layer]

                # Compute Gram matrices
                gen_gram = gram_matrix(gen_feature)
                style_gram = gram_matrix(style_feature)

                # Compute MSE loss between Gram matrices
                loss = nn.functional.mse_loss(gen_gram, style_gram)
                style_loss += loss.item()

            total_style_loss += style_loss
            count += 1

    average_style_loss = total_style_loss / count if count > 0 else 0
    print(f"Average Style Loss: {average_style_loss}")
    return average_style_loss

def compute_ssim_and_show(image1, image2, idx):
    """
    Computes SSIM between two images and displays them side by side.
    """
    # Convert tensors to NumPy arrays and scale to [0, 255]
    image1_np = image1.permute(1, 2, 0).cpu().numpy() * 255.0
    image2_np = image2.permute(1, 2, 0).cpu().numpy() * 255.0
    image1_np = image1_np.astype(np.uint8)
    image2_np = image2_np.astype(np.uint8)

    # Compute SSIM value
    ssim_value = ssim(image1_np, image2_np, multichannel=True, win_size=3)
    print(f"SSIM for image {idx}: {ssim_value}")

    # Display images side by side
    _, axes = plt.subplots(1, 2, figsize=(8, 5))
    axes[0].imshow(image1_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(image2_np)
    axes[1].set_title("Generated Image")
    axes[1].axis('off')

    # Display SSIM value as a subtitle
    plt.suptitle(f"SSIM: {ssim_value:.4f} (Image {idx})", fontsize=16)
    plt.tight_layout()
    plt.show()

    return ssim_value

def evaluate_ssim_over_random_images(test_data_loader, painting_generator, num_images=10):
    """
    Evaluates SSIM on randomly selected images from the test_data_loader.
    """
    # Set the generator model to evaluation mode
    painting_generator.eval()
    ssim_values = []

    with torch.no_grad():
        # Randomly select indices from the test data
        random_indices = random.sample(range(len(test_data_loader.dataset)), num_images)

        for idx, (content_image, _) in enumerate(test_data_loader):
            if idx in random_indices:
                content_image = content_image.to(config.DEVICE)
 
                # Generate painting-style image using the painting generator
                generated_image = painting_generator(content_image)
 
                # Rescale the generated image and content image to [0, 1] for visualization
                generated_image = generated_image * 0.5 + 0.5
                content_image = content_image * 0.5 + 0.5

                # Compute SSIM and show the images
                ssim_value = compute_ssim_and_show(content_image[0], generated_image[0], idx)
                ssim_values.append(ssim_value)

    # Calculate and display the average SSIM across all selected images
    avg_ssim = np.mean(ssim_values)
    print(f"\nAverage SSIM for the selected {num_images} images: {avg_ssim:.4f}")
