"""
Main execution script for MDGAN with 5-fold cross-validation.
"""

import torch
import numpy as np
import os
import gc
import warnings
import traceback

from config import (
    DATA_ROOT_DIR, OUTPUT_ROOT_DIR, CV_FOLDS, TORCH_SEED, NUMPY_SEED,
    LATENT_DIM, LEARNING_RATE, NUM_GENERATORS, NUM_DISCRIMINATORS, NUM_EPOCHS
)
from dataset import validate_datasets, load_and_preprocess_data, create_dataloader
from models import initialize_models, initialize_optimizers
from trainer import train_mdgan, generate_final_samples

# Suppress warnings
warnings.filterwarnings('ignore')


def setup_environment():
    """
    Setup the environment and random seeds.

    Returns:
        torch.device: Device for computation
    """
    # Set random seeds for reproducibility
    torch.manual_seed(TORCH_SEED)
    np.random.seed(NUMPY_SEED)

    # Get computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    return device


def process_single_fold(dataset_folder, cv_fold, dataset_output_dir, device):
    """
    Process a single cross-validation fold.

    Args:
        dataset_folder (str): Path to dataset folder
        cv_fold (str): Name of CV fold (e.g., '1cv')
        dataset_output_dir (str): Output directory for dataset
        device (torch.device): Device for computation

    Returns:
        bool: True if processing was successful, False otherwise
    """
    print(f"\nProcessing fold: {cv_fold}")

    # Setup paths
    cv_fold_path = os.path.join(dataset_folder, cv_fold)
    minority_file = os.path.join(cv_fold_path, 'minority.csv')
    majority_file = os.path.join(cv_fold_path, 'majority.csv')
    fold_output_dir = os.path.join(dataset_output_dir, cv_fold)
    os.makedirs(fold_output_dir, exist_ok=True)

    try:
        # Load and preprocess data
        minority_data, majority_data, scaler = load_and_preprocess_data(minority_file, majority_file)

        print(f"Number of minority samples: {len(minority_data)}")
        print(f"Number of majority samples: {len(majority_data)}")

        # Create dataloader
        minority_dataloader = create_dataloader(minority_data)

        # Get data dimension
        data_dim = minority_data.shape[1]

        # Initialize models
        generators, discriminators = initialize_models(
            NUM_GENERATORS, NUM_DISCRIMINATORS, LATENT_DIM, data_dim, device
        )

        # Initialize optimizers
        optimizers_G, optimizers_D = initialize_optimizers(
            generators, discriminators, LEARNING_RATE
        )

        # Train MDGAN
        losses = train_mdgan(
            minority_dataloader, minority_data, majority_data,
            generators, discriminators, optimizers_G, optimizers_D,
            device, fold_output_dir
        )

        # Generate final samples
        generate_final_samples(
            generators, minority_data, majority_data, device, fold_output_dir, NUM_EPOCHS
        )

        # Free memory
        del minority_data, majority_data, minority_dataloader
        del generators, discriminators, optimizers_G, optimizers_D
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return True

    except Exception as e:
        print(f"Error processing fold {cv_fold}: {str(e)}")
        traceback.print_exc()
        return False


def process_single_dataset(dataset_folder, device):
    """
    Process a single dataset with all its cross-validation folds.

    Args:
        dataset_folder (str): Path to dataset folder
        device (torch.device): Device for computation

    Returns:
        bool: True if processing was successful, False otherwise
    """
    dataset_name = os.path.basename(dataset_folder)
    print(f"\nProcessing dataset: {dataset_name}")

    # Create dataset-specific output directory
    dataset_output_dir = os.path.join(OUTPUT_ROOT_DIR, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    success_count = 0

    # Process each CV fold
    for cv_fold in CV_FOLDS:
        if process_single_fold(dataset_folder, cv_fold, dataset_output_dir, device):
            success_count += 1

    print(f"Successfully processed {success_count}/{len(CV_FOLDS)} folds for dataset {dataset_name}")
    return success_count == len(CV_FOLDS)


def main():
    """
    Main function to execute MDGAN training with 5-fold cross-validation.
    """
    print("=" * 60)
    print("MDGAN Training with 5-Fold Cross-Validation")
    print("=" * 60)

    # Setup environment
    device = setup_environment()

    # Validate datasets
    valid_dataset_folders = validate_datasets(DATA_ROOT_DIR, CV_FOLDS)

    if not valid_dataset_folders:
        print(f"No valid datasets with 5-fold CV structure found in: {DATA_ROOT_DIR}")
        return

    print(f"Found {len(valid_dataset_folders)} valid datasets with 5-fold CV structure")

    # Process each dataset
    successful_datasets = 0

    for dataset_folder in valid_dataset_folders:
        try:
            if process_single_dataset(dataset_folder, device):
                successful_datasets += 1
        except Exception as e:
            dataset_name = os.path.basename(dataset_folder)
            print(f"Failed to process dataset {dataset_name}: {str(e)}")
            traceback.print_exc()

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Successfully processed: {successful_datasets}/{len(valid_dataset_folders)} datasets")
    print(f"Results saved to: {OUTPUT_ROOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()