"""
Utility functions for MDGAN project.
"""

import torch
import numpy as np
import pandas as pd
import os
import gc
from sklearn.preprocessing import MinMaxScaler


def set_random_seeds(torch_seed=42, numpy_seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        torch_seed (int): Seed for PyTorch
        numpy_seed (int): Seed for NumPy
    """
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)


def cleanup_memory():
    """
    Clean up GPU and system memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def check_data_consistency(minority_data, majority_data):
    """
    Check consistency between minority and majority data.

    Args:
        minority_data (numpy.ndarray): Minority class data
        majority_data (numpy.ndarray): Majority class data

    Returns:
        bool: True if data is consistent, False otherwise
    """
    if minority_data.shape[1] != majority_data.shape[1]:
        print(f"Error: Feature dimensions don't match: {minority_data.shape[1]} vs {majority_data.shape[1]}")
        return False

    if len(minority_data) == 0 or len(majority_data) == 0:
        print("Error: One of the datasets is empty")
        return False

    return True


def normalize_data(data, scaler=None, fit_scaler=True):
    """
    Normalize data using MinMaxScaler.

    Args:
        data (numpy.ndarray): Data to normalize
        scaler (MinMaxScaler, optional): Pre-fitted scaler
        fit_scaler (bool): Whether to fit the scaler

    Returns:
        tuple: (normalized_data, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler()

    if fit_scaler:
        normalized_data = scaler.fit_transform(data)
    else:
        normalized_data = scaler.transform(data)

    return normalized_data, scaler


def save_data_statistics(minority_data, majority_data, output_dir, fold_name):
    """
    Save data statistics to a text file.

    Args:
        minority_data (numpy.ndarray): Minority class data
        majority_data (numpy.ndarray): Majority class data
        output_dir (str): Output directory
        fold_name (str): Name of the fold
    """
    stats_file = os.path.join(output_dir, f'data_statistics_{fold_name}.txt')

    with open(stats_file, 'w') as f:
        f.write(f"Data Statistics for {fold_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Minority class samples: {len(minority_data)}\n")
        f.write(f"Majority class samples: {len(majority_data)}\n")
        f.write(f"Feature dimensions: {minority_data.shape[1]}\n")
        f.write(f"Imbalance ratio: {len(majority_data) / len(minority_data):.2f}\n")
        f.write("\nMinority class statistics:\n")
        f.write(f"Mean: {np.mean(minority_data, axis=0)}\n")
        f.write(f"Std: {np.std(minority_data, axis=0)}\n")
        f.write("\nMajority class statistics:\n")
        f.write(f"Mean: {np.mean(majority_data, axis=0)}\n")
        f.write(f"Std: {np.std(majority_data, axis=0)}\n")


def load_generated_samples(output_dir, epoch='final'):
    """
    Load generated samples from CSV files.

    Args:
        output_dir (str): Directory containing generated samples
        epoch (str or int): Epoch identifier

    Returns:
        numpy.ndarray or None: Generated samples array
    """
    filename = os.path.join(output_dir, f'all_generated_samples_epoch_{epoch}.csv')

    if os.path.exists(filename):
        return np.loadtxt(filename, delimiter=',')
    else:
        print(f"Generated samples file not found: {filename}")
        return None


def create_directory_structure(base_dir, dataset_names, cv_folds):
    """
    Create directory structure for multiple datasets and CV folds.

    Args:
        base_dir (str): Base output directory
        dataset_names (list): List of dataset names
        cv_folds (list): List of CV fold names
    """
    os.makedirs(base_dir, exist_ok=True)

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        for cv_fold in cv_folds:
            fold_dir = os.path.join(dataset_dir, cv_fold)
            os.makedirs(fold_dir, exist_ok=True)


def get_device_info():
    """
    Get information about available computing devices.

    Returns:
        dict: Dictionary containing device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }

    return device_info


def validate_file_paths(file_paths):
    """
    Validate that all file paths exist.

    Args:
        file_paths (list): List of file paths to validate

    Returns:
        tuple: (valid_paths, invalid_paths)
    """
    valid_paths = []
    invalid_paths = []

    for path in file_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            invalid_paths.append(path)

    return valid_paths, invalid_paths


def calculate_model_parameters(model):
    """
    Calculate the total number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model

    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def print_model_summary(generators, discriminators):
    """
    Print summary of model architectures and parameters.

    Args:
        generators (list): List of generator models
        discriminators (list): List of discriminator models
    """
    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)

    # Generator summary
    if generators:
        total_g, trainable_g = calculate_model_parameters(generators[0])
        print(f"Generator Architecture:")
        print(f"  - Number of generators: {len(generators)}")
        print(f"  - Parameters per generator: {total_g:,}")
        print(f"  - Trainable parameters: {trainable_g:,}")

    # Discriminator summary
    if discriminators:
        total_d, trainable_d = calculate_model_parameters(discriminators[0])
        print(f"Discriminator Architecture:")
        print(f"  - Number of discriminators: {len(discriminators)}")
        print(f"  - Parameters per discriminator: {total_d:,}")
        print(f"  - Trainable parameters: {trainable_d:,}")

    print("=" * 50)