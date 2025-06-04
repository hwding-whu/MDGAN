"""
Configuration file for MDGAN project.
Contains all hyperparameters and settings.
"""

# Model hyperparameters
LATENT_DIM = 128
NUM_EPOCHS = 2000
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
SAMPLE_INTERVAL = 500
NUM_GENERATORS = 3
NUM_DISCRIMINATORS = 3

# Loss function parameters
MARGIN = 0.5  # Margin parameter for contrastive loss
LAMBDA_GP = 10  # WGAN-GP gradient penalty coefficient
LAMBDA_ME = 0.1  # Mutual exclusion loss weight
LAMBDA_CL = 0.1  # Contrastive loss weight

# WGAN-GP training parameters
N_CRITIC = 5  # Number of discriminator training iterations per generator iteration

# Data paths
DATA_ROOT_DIR = './data'
OUTPUT_ROOT_DIR = './generated/MDGAN-CV'

# Cross-validation folds
CV_FOLDS = ['1cv', '2cv', '3cv', '4cv', '5cv']

# Random seeds
TORCH_SEED = 42
NUMPY_SEED = 42

# Visualization parameters
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 300
MAX_TSNE_SAMPLES = 1000