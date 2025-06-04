# MDGAN: Multiple Discriminator Generative Adversarial Network

MDGAN is a sophisticated generative adversarial network designed for generating synthetic tabular data to address class imbalance problems. It employs multiple generators and discriminators with specialized loss functions including mutual exclusion loss and contrastive loss.

## Features

- **Multiple Generators & Discriminators**: Uses multiple generator-discriminator pairs for improved diversity
- **WGAN-GP Training**: Implements Wasserstein GAN with Gradient Penalty for stable training
- **Mutual Exclusion Loss**: Encourages generators to produce diverse outputs
- **Contrastive Loss**: Guides generated samples to be similar to minority class and different from majority class
- **Cross-Validation Support**: Built-in support for 5-fold cross-validation
- **Memory Optimization**: Includes memory management and garbage collection for large datasets
- **Comprehensive Visualization**: t-SNE plots and loss curve monitoring

## Project Structure

```
MDGAN/
├── config.py              # Configuration and hyperparameters
├── dataset.py             # Dataset loading and preprocessing
├── models.py              # Generator and Discriminator architectures
├── losses.py              # Loss functions (gradient penalty, mutual exclusion, contrastive)
├── trainer.py             # Training logic and model management
├── visualization.py       # Sample generation and visualization utilities
├── utils.py               # Utility functions
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

1. **Clone the repository** (if applicable) or save all Python files in a single directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify CUDA installation** (optional but recommended for GPU acceleration):
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## Data Format

The expected data structure should be organized as follows:

```
../data/
├── dataset1/
│   ├── 1cv/
│   │   ├── minority.csv
│   │   └── majority.csv
│   ├── 2cv/
│   │   ├── minority.csv
│   │   └── majority.csv
│   ├── ... (3cv, 4cv, 5cv)
├── dataset2/
│   └── ... (same structure)
└── ...
```

- **CSV files should contain numerical data only (no headers)**
- **Each row represents a sample**
- **Each column represents a feature**
- **All features should be numerical**

## Configuration

Key hyperparameters can be modified in `config.py`:

```python
# Model hyperparameters
LATENT_DIM = 128                # Dimension of noise vector
NUM_EPOCHS = 2000              # Number of training epochs
BATCH_SIZE = 32                # Batch size for training
LEARNING_RATE = 0.0002         # Learning rate for optimizers
NUM_GENERATORS = 3             # Number of generators
NUM_DISCRIMINATORS = 3         # Number of discriminators

# Loss function parameters
LAMBDA_GP = 10                 # WGAN-GP gradient penalty coefficient
LAMBDA_ME = 0.1               # Mutual exclusion loss weight
LAMBDA_CL = 0.1               # Contrastive loss weight
MARGIN = 0.5                  # Margin for contrastive loss

# Training parameters
N_CRITIC = 5                  # Discriminator updates per generator update
SAMPLE_INTERVAL = 500         # Epoch interval for saving samples
```

## Usage

### Basic Usage

Run the main training script:

```bash
python main.py
```

This will:
1. Automatically detect all valid datasets in the `../data/` directory
2. Process each dataset with 5-fold cross-validation
3. Train MDGAN models for each fold
4. Generate synthetic samples and visualizations
5. Save results in `./generated/MDGAN-CV/`

### Custom Configuration

1. **Modify data paths** in `config.py`:
   ```python
   DATA_ROOT_DIR = '/path/to/your/data'
   OUTPUT_ROOT_DIR = '/path/to/output'
   ```

2. **Adjust hyperparameters** in `config.py` based on your dataset characteristics

3. **Modify model architectures** in `models.py` if needed

### Advanced Usage