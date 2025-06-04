"""
Neural network models for MDGAN.
Contains Generator and Discriminator architectures.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for MDGAN.
    Transforms random noise into synthetic tabular data.
    """

    def __init__(self, z_dim, data_dim):
        """
        Initialize the Generator.

        Args:
            z_dim (int): Dimension of input noise vector
            data_dim (int): Dimension of output data
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, data_dim),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Random noise vector

        Returns:
            torch.Tensor: Generated synthetic data
        """
        return self.main(z)


class Discriminator(nn.Module):
    """
    Discriminator network for MDGAN using WGAN-GP style.
    Distinguishes between real and synthetic data.
    """

    def __init__(self, data_dim):
        """
        Initialize the Discriminator.

        Args:
            data_dim (int): Dimension of input data
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)  # WGAN doesn't use sigmoid
        )

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Discriminator output score
        """
        return self.main(x)


def initialize_models(num_generators, num_discriminators, latent_dim, data_dim, device):
    """
    Initialize multiple generators and discriminators.

    Args:
        num_generators (int): Number of generators to create
        num_discriminators (int): Number of discriminators to create
        latent_dim (int): Dimension of latent space
        data_dim (int): Dimension of data
        device (torch.device): Device to place models on

    Returns:
        tuple: (generators, discriminators)
    """
    generators = [Generator(latent_dim, data_dim).to(device) for _ in range(num_generators)]
    discriminators = [Discriminator(data_dim).to(device) for _ in range(num_discriminators)]

    return generators, discriminators


def initialize_optimizers(generators, discriminators, learning_rate):
    """
    Initialize optimizers for generators and discriminators.

    Args:
        generators (list): List of generator models
        discriminators (list): List of discriminator models
        learning_rate (float): Learning rate for optimizers

    Returns:
        tuple: (optimizers_G, optimizers_D)
    """
    optimizers_G = [
        torch.optim.Adam(g.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        for g in generators
    ]
    optimizers_D = [
        torch.optim.Adam(d.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        for d in discriminators
    ]

    return optimizers_G, optimizers_D