# Balancing Act: MDGAN for Imbalanced Tabular Data Synthesis

Addressing the persistent challenge of learning from imbalanced datasets is crucial in advancing machine learning applications. Standard machine learning algorithms typically assume that the input data is balanced, and they often struggle to effectively learn the distribution of minority class data when dealing with imbalanced data. To address this, our study designed an improved Generative Adversarial Networks (GANs) model, named MDGAN, for tabular sample synthesis to augment samples and balance the data distribution. MDGAN employs a multi-generator and multi-discriminator structure to capture non-connected subspace manifolds, thereby better fitting the complete data distribution. To enhance the diversity among the multiple generators, an exclusive loss among generators was designed, ensuring that each generator produces data of different modalities. Additionally, a contrastive loss was introduced to ensure that the generated samples better fit the minority class distribution and are separated from the majority class distribution, preventing blurred classification boundaries. Qualitative and quantitative tests were conducted on 25 real datasets, and the experimental results indicate that MDGAN outperforms traditional classical models and current advanced oversampling models.

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
