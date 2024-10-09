"""
This module contains the training function for the CycleGAN model.
"""
import torch

from torchvision.utils import save_image
from tqdm import tqdm
import config

def train_fn(
    disc_photo,
    disc_paint,
    gen_paint,
    gen_photo,
    loader,
    opt_disc,
    opt_gen,
    l1,
    bce,
    d_scaler,
    g_scaler
):
    """
    Trains the discriminator and generator networks for photo and paint images.

    Args:
        disc_photo (torch.nn.Module): Discriminator network for photo images.
        disc_paint (torch.nn.Module): Discriminator network for paint images.
        gen_paint (torch.nn.Module): Generator network for paint images.
        gen_photo (torch.nn.Module): Generator network for photo images.
        loader (torch.utils.data.DataLoader): DataLoader for loading the training data.
        opt_disc (torch.optim.Optimizer): Optimizer for the discriminator networks.
        opt_gen (torch.optim.Optimizer): Optimizer for the generator networks.
        l1 (torch.nn.Module): L1 loss function.
        mse (torch.nn.Module): Mean Squared Error (MSE) loss function.
        d_scaler (torch.cuda.amp.GradScaler): Gradient scaler for discriminator networks.
        g_scaler (torch.cuda.amp.GradScaler): Gradient scaler for generator networks.

    Returns:
        None
    """
    photo_reals = 0
    photo_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (paint, photo) in enumerate(loop):
        # Move to GPU
        paint = paint.to(config.DEVICE)
        photo = photo.to(config.DEVICE)

        # Train Discriminators photo and paint
        with torch.amp.autocast('cuda'):
            # Generate fake photo image
            fake_photo = gen_photo(paint)
            # Discriminate real and fake photo images
            d_photo_real = disc_photo(photo)
            d_photo_fake = disc_photo(fake_photo.detach()) # detach to prevent backpropagation
            # Calculate loss
            photo_reals += d_photo_real.mean().item()
            photo_fakes += d_photo_fake.mean().item()
            d_photo_real_loss = bce(d_photo_real, torch.ones_like(d_photo_real))
            d_photo_fake_loss = bce(d_photo_fake, torch.zeros_like(d_photo_fake))
            d_photo_loss = d_photo_real_loss + d_photo_fake_loss
            # Generate fake paint image
            fake_paint = gen_paint(photo)
            # Discriminate real and fake paint images
            d_paint_real = disc_paint(paint)
            d_paint_fake = disc_paint(fake_paint.detach())
            # Calculate loss
            d_paint_real_loss = bce(d_paint_real, torch.ones_like(d_paint_real))
            d_paint_fake_loss = bce(d_paint_fake, torch.zeros_like(d_paint_fake))
            d_paint_loss = d_paint_real_loss + d_paint_fake_loss

            # Combine losses
            d_loss = (d_photo_loss + d_paint_loss) / 2

        # Backpropagation
        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators photo and paint
        with torch.amp.autocast('cuda'):
            # Calculate adversarial loss
            d_photo_fake = disc_photo(fake_photo)
            d_paint_fake = disc_paint(fake_paint)
            loss_g_photo = bce(d_photo_fake, torch.ones_like(d_photo_fake))
            loss_g_paint = bce(d_paint_fake, torch.ones_like(d_paint_fake))
            # Calculate cycle loss
            cycle_paint = gen_paint(fake_photo)
            cycle_photo = gen_photo(fake_paint)
            cycle_paint_loss = l1(paint, cycle_paint)
            cycle_photo_loss = l1(photo, cycle_photo)
            # Calculate identity loss
            identity_paint = gen_paint(paint)
            identity_photo = gen_photo(photo)
            identity_paint_loss = l1(paint, identity_paint)
            identity_photo_loss = l1(photo, identity_photo)

            # Combine losses
            g_loss = (
                loss_g_paint
                + loss_g_photo
                + cycle_paint_loss * config.LAMBDA_CYCLE
                + cycle_photo_loss * config.LAMBDA_CYCLE
                + identity_photo_loss * config.LAMBDA_IDENTITY
                + identity_paint_loss * config.LAMBDA_IDENTITY
            )

        # Backpropagation
        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_photo * 0.5 + 0.5, f"saved_images/photo_{idx}.png", format='png')
            save_image(fake_paint * 0.5 + 0.5, f"saved_images/paint_{idx}.png", format='png')

        loop.set_postfix(
            photo_real=photo_reals / (idx + 1),
            photo_fake=photo_fakes / (idx + 1),
            g_loss=g_loss.item(),
            d_loss=d_loss.item(),
        )
