"""
Module containing utility functions.
"""
import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import config

def denormalize(*args):
    """
    Denormalizes the given arguments by scaling them by 0.5 and then adding 0.5.

    Parameters:
    *args: Variable length argument list of numerical values to be denormalized.

    Returns:
    list: A list of denormalized values.
    """
    return [arg * 0.5 + 0.5 for arg in args]

def save_epoch_loss_results(epoch, losses, iteration):
    """
    Saves a plot of generator and discriminator losses for a given epoch.
    Args:
        epoch (int): The current epoch number.
        losses (tuple): A tuple containing two lists:
            - generator_losses (list): List of generator loss values.
            - discriminator_losses (list): List of discriminator loss values.
    """
    generator_losses, discriminator_losses = losses
    steps = [i for i, _ in enumerate(generator_losses)]

    plt.plot(steps, generator_losses, label="Generator Loss", color="blue")
    plt.plot(steps, discriminator_losses, label="Discriminator Loss", color="red")

    plt.grid(True)

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Generator and Discriminator Losses for Epoch {epoch+1}")

    if not os.path.exists(config.LOSS_PLOTS_DIR):
        os.makedirs(config.LOSS_PLOTS_DIR)
        
    if not os.path.exists(os.path.join(config.LOSS_PLOTS_DIR, f"iteration_{str(iteration)}")):
        os.makedirs(os.path.join(config.LOSS_PLOTS_DIR, f"iteration_{str(iteration)}"))

    plt.savefig(os.path.join(config.LOSS_PLOTS_DIR, f"iteration_{str(iteration)}/epoch_{epoch+1}_losses.png"))
    plt.close()

def generate_test_images(
    painting_generator,
    content_generator,
    test_data_loader,
    generated_content_path,
    generated_paintings_path,
):
    """
    Generate and save test images using the provided painting and content generators.

    Args:
        painting_generator (torch.nn.Module): The model used to generate painting-style images from content images.
        content_generator (torch.nn.Module): The model used to generate content-style images from painting images.
        test_data_loader (torch.utils.data.DataLoader): DataLoader providing batches of content and painting images for testing.
        generated_content_path (str): Directory path to save the generated content-style images.
        generated_paintings_path (str): Directory path to save the generated painting-style images.

    Returns:
        None
    """
    # Create directories to save the generated images
    os.makedirs(generated_content_path, exist_ok=True)
    os.makedirs(generated_paintings_path, exist_ok=True)

    # Set the models to evaluation mode
    painting_generator.eval()
    content_generator.eval()

    with torch.no_grad():
        for idx, (content_images, painting_images) in enumerate(tqdm(test_data_loader)):
            content_images = content_images.to(config.DEVICE)
            painting_images = painting_images.to(config.DEVICE)

            # Generate images
            fake_painting = painting_generator(content_images)
            fake_content = content_generator(painting_images)

            # Rescale the outputs to [0, 1]
            fake_painting = fake_painting * 0.5 + 0.5
            fake_content = fake_content * 0.5 + 0.5

            # Save the generated images for each model
            save_image(fake_painting, f'{generated_paintings_path}/generated_{idx+1}.png')
            save_image(fake_content, f'{generated_content_path}/generated_{idx+1}.png')

def preprocess_images(
    input_dir,
    output_dir,
    image_height=config.IMAGE_HEIGHT,
    image_width=config.IMAGE_WIDTH
):
    """
    Resizes and normalizes images in input_dir and saves them to output_dir.

    Args:
        input_dir (str): Path to the directory containing original images.
        output_dir (str): Path to save preprocessed images.
        image_height (int): Height for resizing.
        image_width (int): Width for resizing.
    """
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
    ])

    # Iterate through images with progress bar
    for img_name in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name)
        with Image.open(img_path) as img:
            # Ensure the image is in RGB mode
            img = img.convert("RGB")
            img = transform(img)
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(output_dir, img_name))

    print(f"Preprocessed images saved to {output_dir}")

def gram_matrix(features):
    """
    Computes the Gram matrix from features.
    """
    batch_size, channels, height, width = features.size()
    # Flatten the feature maps
    features = features.view(batch_size, channels, height * width)
    # Compute the Gram matrix
    G = torch.bmm(features, features.transpose(1, 2))
    # Normalize the Gram matrix
    G = G / (channels * height * width)
    return G
