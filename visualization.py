"""
Visualization and sample generation utilities for MDGAN.
"""

import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import math
import gc
from config import BATCH_SIZE, LATENT_DIM, TSNE_PERPLEXITY, TSNE_N_ITER, MAX_TSNE_SAMPLES


def generate_samples(generators, num_samples_to_generate, latent_dim, device, batch_size=BATCH_SIZE):
    """
    Generate synthetic samples using multiple generators.

    Args:
        generators (list): List of generator models
        num_samples_to_generate (int): Total number of samples to generate
        latent_dim (int): Dimension of latent space
        device (torch.device): Device for computation
        batch_size (int): Batch size for generation

    Returns:
        tuple: (all_generated_samples, generator_labels)
    """
    samples_per_generator = math.ceil(num_samples_to_generate / len(generators))
    all_generated_samples = []
    generator_labels = []

    with torch.no_grad():
        for g_idx, generator in enumerate(generators):
            generator.eval()
            generated_samples = []

            remaining = samples_per_generator
            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                z = torch.randn(current_batch_size, latent_dim, device=device)
                fake = generator(z).cpu().numpy()
                generated_samples.extend(fake)
                remaining -= current_batch_size

                # Free memory
                del z, fake
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Truncate to required number of samples per generator
            generated_samples = generated_samples[:samples_per_generator]
            all_generated_samples.extend(generated_samples)
            generator_labels.extend([g_idx] * len(generated_samples))

    # Truncate to the required total number of samples
    all_generated_samples = all_generated_samples[:num_samples_to_generate]
    generator_labels = generator_labels[:num_samples_to_generate]

    return all_generated_samples, generator_labels


def save_generated_samples(generators, output_dir, epoch, num_samples, latent_dim, device):
    """
    Save generated samples from each generator to CSV files.

    Args:
        generators (list): List of generator models
        output_dir (str): Output directory
        epoch (int or str): Current epoch number
        num_samples (int): Number of samples to generate per generator
        latent_dim (int): Dimension of latent space
        device (torch.device): Device for computation
    """
    samples_per_generator = math.ceil(num_samples / len(generators))

    with torch.no_grad():
        for g_idx, generator in enumerate(generators):
            generator.eval()
            generated_samples = []

            remaining = samples_per_generator
            while remaining > 0:
                current_batch_size = min(BATCH_SIZE, remaining)
                z = torch.randn(current_batch_size, latent_dim, device=device)
                fake = generator(z).cpu().numpy()
                generated_samples.extend(fake)
                remaining -= current_batch_size

                # Free memory
                del z, fake
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save samples
            generator_samples_np = np.array(generated_samples[:samples_per_generator])
            np.savetxt(
                os.path.join(output_dir, f'generator_{g_idx}_samples_epoch_{epoch}.csv'),
                generator_samples_np, delimiter=','
            )


def create_tsne_visualization(all_data, viz_labels, generators, output_dir, epoch):
    """
    Create t-SNE visualization of real and generated data.

    Args:
        all_data (numpy.ndarray): Combined real and generated data
        viz_labels (numpy.ndarray): Labels for visualization
        generators (list): List of generators
        output_dir (str): Output directory
        epoch (int or str): Current epoch
    """
    n_samples = len(all_data)

    if n_samples <= 20:
        print("Too few samples for t-SNE visualization")
        return

    # Downsample if dataset is too large
    if n_samples > MAX_TSNE_SAMPLES:
        sample_indices = np.random.choice(n_samples, size=MAX_TSNE_SAMPLES, replace=False)
        all_data_sample = all_data[sample_indices]
        viz_labels_sample = viz_labels[sample_indices]
    else:
        all_data_sample = all_data
        viz_labels_sample = viz_labels

    perplexity = min(TSNE_PERPLEXITY, len(all_data_sample) - 1)

    try:
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=TSNE_N_ITER, random_state=42)
        tsne_results = tsne.fit_transform(all_data_sample)

        # Create visualization
        plt.figure(figsize=(10, 8))

        # Plot real minority class data
        plt.scatter(
            tsne_results[viz_labels_sample == 0, 0],
            tsne_results[viz_labels_sample == 0, 1],
            c='blue', label='Real Minority', alpha=0.7, s=40
        )

        # Plot samples generated by each generator
        colors = ['green', 'purple', 'orange', 'cyan', 'magenta']
        for g_idx in range(len(generators)):
            mask = viz_labels_sample == g_idx + 1
            if np.any(mask):
                plt.scatter(
                    tsne_results[mask, 0],
                    tsne_results[mask, 1],
                    c=colors[g_idx % len(colors)],
                    label=f'Generator {g_idx}',
                    alpha=0.5, s=40
                )

        plt.title(f't-SNE Visualization (Minority & Generated) at Epoch {epoch}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tsne_epoch_{epoch}.png'))
        plt.close()

        # Free memory
        del tsne_results, all_data_sample, viz_labels_sample

    except Exception as e:
        print(f"t-SNE visualization failed: {str(e)}")


def generate_and_visualize(generators, minority_data, majority_data, device, output_dir, epoch):
    """
    Generate samples and create visualizations.

    Args:
        generators (list): List of generator models
        minority_data (numpy.ndarray): Minority class data
        majority_data (numpy.ndarray): Majority class data
        device (torch.device): Device for computation
        output_dir (str): Output directory
        epoch (int or str): Current epoch
    """
    num_samples_to_generate = len(minority_data)

    # Generate samples
    all_generated_samples, generator_labels = generate_samples(
        generators, num_samples_to_generate, LATENT_DIM, device
    )

    # Save individual generator samples
    save_generated_samples(generators, output_dir, epoch, num_samples_to_generate, LATENT_DIM, device)

    # Combine data for visualization (only minority class and generated samples)
    real_data = minority_data
    all_data = np.vstack((real_data, all_generated_samples))

    # Create labels for visualization
    viz_labels = np.concatenate([
        np.zeros(len(real_data)),  # Real minority class label (0)
        np.array(generator_labels) + 1  # Generated sample labels (1, 2, 3, ...)
    ])

    # Create t-SNE visualization
    create_tsne_visualization(all_data, viz_labels, generators, output_dir, epoch)

    # Save all generated samples
    all_generated_samples_np = np.array(all_generated_samples)
    np.savetxt(
        os.path.join(output_dir, f'all_generated_samples_epoch_{epoch}.csv'),
        all_generated_samples_np, delimiter=','
    )

    # Free memory
    del all_generated_samples, all_generated_samples_np, generator_labels, all_data, viz_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot_loss_curves(d_losses, g_losses, me_losses, cl_losses, output_dir):
    """
    Plot and save loss curves.

    Args:
        d_losses (list): Discriminator losses
        g_losses (list): Generator losses
        me_losses (list): Mutual exclusion losses
        cl_losses (list): Contrastive losses
        output_dir (str): Output directory
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(g_losses, label='Generator Adversarial Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Adversarial Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(me_losses, label='Mutual Exclusion Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mutual Exclusion Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(cl_losses, label='Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contrastive Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()