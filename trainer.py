"""
Training logic for MDGAN.
"""

import torch
import os
import gc
from tqdm import tqdm

from config import (
    NUM_EPOCHS, SAMPLE_INTERVAL, LATENT_DIM, N_CRITIC,
    LAMBDA_GP, LAMBDA_ME, LAMBDA_CL, MARGIN
)
from losses import calculate_discriminator_loss, calculate_generator_loss
from visualization import generate_and_visualize, plot_loss_curves


def save_checkpoint(generators, discriminators, epoch, output_dir):
    """
    Save model checkpoint.

    Args:
        generators (list): List of generator models
        discriminators (list): List of discriminator models
        epoch (int): Current epoch
        output_dir (str): Output directory
    """
    checkpoint = {
        'epoch': epoch,
        'generators_state_dict': [g.state_dict() for g in generators],
        'discriminators_state_dict': [d.state_dict() for d in discriminators]
    }
    torch.save(checkpoint, os.path.join(output_dir, f'models_{epoch}.pth'))


def load_checkpoint(generators, discriminators, checkpoint_path):
    """
    Load model checkpoint.

    Args:
        generators (list): List of generator models
        discriminators (list): List of discriminator models
        checkpoint_path (str): Path to checkpoint file

    Returns:
        int: Loaded epoch number
    """
    checkpoint = torch.load(checkpoint_path)

    for g_idx, generator in enumerate(generators):
        generator.load_state_dict(checkpoint['generators_state_dict'][g_idx])

    for d_idx, discriminator in enumerate(discriminators):
        discriminator.load_state_dict(checkpoint['discriminators_state_dict'][d_idx])

    return checkpoint['epoch']


def train_discriminators(discriminators, optimizers_D, real_imgs, generators, device):
    """
    Train discriminators for one iteration.

    Args:
        discriminators (list): List of discriminator models
        optimizers_D (list): List of discriminator optimizers
        real_imgs (torch.Tensor): Real data batch
        generators (list): List of generator models
        device (torch.device): Device for computation

    Returns:
        float: Average discriminator loss
    """
    batch_size = real_imgs.size(0)
    total_d_loss = 0.0

    for d_idx, (discriminator, optimizer_D) in enumerate(zip(discriminators, optimizers_D)):
        for _ in range(N_CRITIC):
            optimizer_D.zero_grad()

            # Generate corresponding fake samples
            g_idx = d_idx % len(generators)
            generator = generators[g_idx]
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_imgs = generator(z).detach()  # Detach computational graph

            # Calculate discriminator loss
            d_loss = calculate_discriminator_loss(
                discriminator, real_imgs, fake_imgs, device, LAMBDA_GP
            )

            # Update discriminator
            d_loss.backward()
            optimizer_D.step()

            total_d_loss += d_loss.item() / (len(discriminators) * N_CRITIC)

            # Free memory
            del d_loss, fake_imgs, z

    return total_d_loss


def train_generators(generators, optimizers_G, discriminators, minority_tensor, majority_tensor, device):
    """
    Train generators for one iteration.

    Args:
        generators (list): List of generator models
        optimizers_G (list): List of generator optimizers
        discriminators (list): List of discriminator models
        minority_tensor (torch.Tensor): Minority class data
        majority_tensor (torch.Tensor): Majority class data
        device (torch.device): Device for computation

    Returns:
        tuple: (total_g_loss, total_me_loss, total_cl_loss)
    """
    total_g_loss = 0.0
    total_me_loss = 0.0
    total_cl_loss = 0.0

    for g_idx, (generator, optimizer_G) in enumerate(zip(generators, optimizers_G)):
        optimizer_G.zero_grad()

        # Calculate generator losses
        discriminator = discriminators[g_idx % len(discriminators)]
        g_loss_total, g_loss_adv, me_loss, cl_loss = calculate_generator_loss(
            generator, discriminator, minority_tensor, majority_tensor,
            generators, LATENT_DIM, device, LAMBDA_ME, LAMBDA_CL, MARGIN
        )

        # Update generator
        g_loss_total.backward()
        optimizer_G.step()

        # Accumulate losses
        total_g_loss += g_loss_adv.item() / len(generators)
        total_me_loss += me_loss.item() / len(generators)
        total_cl_loss += cl_loss.item() / len(generators)

        # Free memory
        del g_loss_total, g_loss_adv, me_loss, cl_loss

    return total_g_loss, total_me_loss, total_cl_loss


def train_mdgan(minority_dataloader, minority_data, majority_data, generators, discriminators,
                optimizers_G, optimizers_D, device, output_dir):
    """
    Main training loop for MDGAN using WGAN-GP training method with memory optimization.

    Args:
        minority_dataloader (DataLoader): DataLoader for minority class data
        minority_data (numpy.ndarray): Minority class data array
        majority_data (numpy.ndarray): Majority class data array
        generators (list): List of generator models
        discriminators (list): List of discriminator models
        optimizers_G (list): List of generator optimizers
        optimizers_D (list): List of discriminator optimizers
        device (torch.device): Device for computation
        output_dir (str): Output directory for saving results

    Returns:
        tuple: (d_losses, g_losses, me_losses, cl_losses, generators)
    """
    # Initialize loss tracking
    d_losses = []
    g_losses = []
    me_losses = []  # Mutual exclusion loss
    cl_losses = []  # Contrastive loss

    # Transfer majority and minority data to device
    minority_tensor = torch.tensor(minority_data, dtype=torch.float32).to(device)
    majority_tensor = torch.tensor(majority_data, dtype=torch.float32).to(device)

    print(f"Starting MDGAN training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_me_loss = 0
        epoch_cl_loss = 0

        # Training loop for current epoch
        for i, real_imgs in enumerate(minority_dataloader):
            real_imgs = real_imgs.to(device)

            # Train discriminators
            d_loss = train_discriminators(
                discriminators, optimizers_D, real_imgs, generators, device
            )
            epoch_d_loss += d_loss

            # Train generators
            g_loss, me_loss, cl_loss = train_generators(
                generators, optimizers_G, discriminators,
                minority_tensor, majority_tensor, device
            )
            epoch_g_loss += g_loss
            epoch_me_loss += me_loss
            epoch_cl_loss += cl_loss

            # Force garbage collection at the end of each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Record average loss for each epoch
        d_losses.append(epoch_d_loss / len(minority_dataloader))
        g_losses.append(epoch_g_loss / len(minority_dataloader))
        me_losses.append(epoch_me_loss / len(minority_dataloader))
        cl_losses.append(epoch_cl_loss / len(minority_dataloader))

        # Print progress and save checkpoints
        if epoch % SAMPLE_INTERVAL == 0 or epoch + 1 == NUM_EPOCHS:
            print(f"[Epoch {epoch}/{NUM_EPOCHS}] "
                  f"[D loss: {d_losses[-1]:.4f}] "
                  f"[G loss: {g_losses[-1]:.4f}] "
                  f"[ME loss: {me_losses[-1]:.4f}] "
                  f"[CL loss: {cl_losses[-1]:.4f}]")

            # Save model checkpoint
            save_checkpoint(generators, discriminators, epoch, output_dir)

            # Generate sample visualization
            generate_and_visualize(generators, minority_data, majority_data, device, output_dir, epoch)

            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Plot and save loss curves
    plot_loss_curves(d_losses, g_losses, me_losses, cl_losses, output_dir)

    return d_losses, g_losses, me_losses, cl_losses, generators


def generate_final_samples(generators, minority_data, majority_data, device, output_dir, num_epochs):
    """
    Generate final samples using trained models.

    Args:
        generators (list): List of trained generator models
        minority_data (numpy.ndarray): Minority class data
        majority_data (numpy.ndarray): Majority class data
        device (torch.device): Device for computation
        output_dir (str): Output directory
        num_epochs (int): Total number of training epochs
    """
    print("Generating final samples...")

    try:
        # Load the last saved model
        checkpoint_path = os.path.join(output_dir, f'models_{num_epochs - 1}.pth')
        if os.path.exists(checkpoint_path):
            load_checkpoint(generators, [], checkpoint_path)

        # Set generators to evaluation mode
        for generator in generators:
            generator.eval()

        # Generate and visualize final samples
        generate_and_visualize(
            generators, minority_data, majority_data,
            device, output_dir, "final"
        )

    except Exception as e:
        print(f"Error loading final model: {str(e)}")
        # Use the current trained model
        for generator in generators:
            generator.eval()
        generate_and_visualize(
            generators, minority_data, majority_data,
            device, output_dir, "final"
        )