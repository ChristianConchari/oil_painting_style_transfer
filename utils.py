"""
Module containing utility functions.
"""
import os
import matplotlib.pyplot as plt
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

def save_epoch_loss_results(epoch, losses):
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
    
    plt.savefig(os.path.join(config.LOSS_PLOTS_DIR, f"epoch_{epoch+1}_losses.png"))
    plt.close()