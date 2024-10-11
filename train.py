"""
This module contains the training function for the CycleGAN model.
"""
import os
import torch
from torchvision.utils import save_image
import config
from tqdm import tqdm as tdqm
import utils

def train_fn(
    content_discriminator,
    painting_discriminator,
    content_generator,
    painting_generator,
    data_loader,
    lambda_cycle=config.LAMBDA_CYCLE,
    lambda_identity=config.LAMBDA_IDENTITY,
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
        bce (torch.nn.Module): Binary Cross-Entropy loss function.
        d_scaler (torch.cuda.amp.GradScaler): Gradient scaler for discriminator networks.
        g_scaler (torch.cuda.amp.GradScaler): Gradient scaler for generator networks.

    Returns:
        None
    """
    # Initialize gradient scalers for mixed precision training
    scaler_gen = torch.amp.GradScaler('cuda')
    scaler_disc = torch.amp.GradScaler('cuda')
    
    # Define the optimizers 
    discriminator_optimizer = torch.optim.Adam(
        list(content_discriminator.parameters()) + list(painting_discriminator.parameters()),
        lr=config.DISC_LEARNING_RATE,
        betas=(config.BETA_1, config.BETA_2),
    )
    generator_optimizer = torch.optim.Adam(
        list(content_generator.parameters()) + list(painting_generator.parameters()),
        lr=config.GEN_LEARNING_RATE,
        betas=(config.BETA_1, config.BETA_2),
    )
    
    print(f'Train on {config.DEVICE}')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch {epoch + 1}/{config.NUM_EPOCHS}')
        gen_losses, disc_losses = [], []
        
        if 0 < epoch < 10:
            discriminator_optimizer.param_groups[0]['lr'] = config.DISC_LEARNING_RATE * 0.5
            generator_optimizer.param_groups[0]['lr'] = config.GEN_LEARNING_RATE * 0.5  
            
        for idx, (content, painting) in enumerate(tdqm(data_loader)):
            content = content.to(config.DEVICE)
            painting = painting.to(config.DEVICE)
            
            # Train the discriminator networks
            with torch.amp.autocast(config.DEVICE):
                # Content discriminator
                fake_content = content_generator(painting)
                disc_content_real = content_discriminator(content)
                content_discriminator_output = content_discriminator(fake_content.detach())
                
                content_discriminator_real_loss = config.MSE_LOSS(disc_content_real, torch.ones_like(disc_content_real))
                content_discriminator_fake_loss = config.MSE_LOSS(content_discriminator_output, torch.zeros_like(content_discriminator_output))
                
                content_discriminator_loss = content_discriminator_real_loss + content_discriminator_fake_loss
                
                # Painting discriminator
                fake_painting = painting_generator(content)
                disc_painting_real = painting_discriminator(painting)
                painting_discriminator_output = painting_discriminator(fake_painting.detach())
                
                painting_discriminator_real_loss = config.MSE_LOSS(disc_painting_real, torch.ones_like(disc_painting_real))
                painting_discriminator_fake_loss = config.MSE_LOSS(painting_discriminator_output, torch.zeros_like(painting_discriminator_output))
                
                painting_discriminator_loss = painting_discriminator_real_loss + painting_discriminator_fake_loss
                
                # Combine discriminator losses
                discriminator_loss = (content_discriminator_loss + painting_discriminator_loss) / 2
            
            # Backpropagation for discriminator networks
            discriminator_optimizer.zero_grad()
            scaler_disc.scale(discriminator_loss).backward()
            scaler_disc.step(discriminator_optimizer)
            scaler_disc.update()
            
            # Train the generator networks
            with torch.amp.autocast(config.DEVICE):
                # Adversarial loss for generators
                content_discriminator_output = content_discriminator(fake_content)
                painting_discriminator_output = painting_discriminator(fake_painting)
                
                fake_content_logits = torch.ones_like(content_discriminator_output) * 0.9
                fake_painting_logits = torch.ones_like(painting_discriminator_output) * 0.9
                
                adversarial_photo_loss = config.MSE_LOSS(content_discriminator_output, fake_content_logits)
                adversarial_painting_loss = config.MSE_LOSS(painting_discriminator_output, fake_painting_logits)
                
                # Cycle consistency loss
                cycle_content_loss = config.L1_LOSS(content, content_generator(fake_painting))
                cycle_paintings_loss = config.L1_LOSS(painting, painting_generator(fake_content))
                
                # Identity loss
                identity_content_loss = config.L1_LOSS(content, content_generator(content))
                identity_painting_loss = config.L1_LOSS(painting, painting_generator(painting))
                
                # Combine generator losses
                generator_loss = (
                    adversarial_photo_loss
                    + adversarial_painting_loss
                    + cycle_content_loss * lambda_cycle
                    + cycle_paintings_loss * lambda_cycle
                    + identity_content_loss * lambda_identity
                    + identity_painting_loss * lambda_identity
                )
                
            # Backpropagation for generator networks
            generator_optimizer.zero_grad()
            scaler_gen.scale(generator_loss).backward()
            scaler_gen.step(generator_optimizer)
            scaler_gen.update()
            
            gen_losses.append(generator_loss.item())
            disc_losses.append(discriminator_loss.item())
            
            if idx % config.VALIDATION_STEP == 0:
                real_painting, generated_content, real_content, generated_painting = utils.denormalize(painting, fake_content, content, fake_painting)
                
                if not os.path.exists(config.CONTENT_RESULTS_DIR):
                    os.makedirs(config.CONTENT_RESULTS_DIR)
                
                save_image(
                    torch.cat([
                        real_painting,
                        generated_content
                        ], dim=3),
                        f"{config.CONTENT_RESULTS_DIR}/content_{epoch+1}_{idx}.png"
                    )

                if not os.path.exists(config.PAINTINGS_RESULTS_DIR):
                    os.makedirs(config.PAINTINGS_RESULTS_DIR)
                
                save_image(
                    torch.cat([
                        real_content,
                        generated_painting
                        ], dim=3),
                        f"{config.PAINTINGS_RESULTS_DIR}/paintings_{epoch+1}_{idx}.png"
                    )
            
        print(f'Mean Generator Loss: {torch.tensor(gen_losses, dtype=torch.float32).mean()}\n')
        print(f'Mean Discriminator Loss: {torch.tensor(disc_losses, dtype=torch.float32).mean()}\n')
        
        utils.save_epoch_loss_results(epoch, [gen_losses, disc_losses])
