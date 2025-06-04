"""
Dataset class and data loading utilities for MDGAN.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from config import BATCH_SIZE


class TabularDataset(Dataset):
    """
    Custom dataset class for tabular data.
    """

    def __init__(self, data):
        """
        Initialize the dataset.

        Args:
            data (numpy.ndarray): Input data array
        """
        self.data = data
        self.data_dim = self.data.shape[1]

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            torch.Tensor: Sample as a float tensor
        """
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)


def load_and_preprocess_data(minority_file, majority_file):
    """
    Load and preprocess minority and majority class data.

    Args:
        minority_file (str): Path to minority class CSV file
        majority_file (str): Path to majority class CSV file

    Returns:
        tuple: (minority_data, majority_data, scaler)
    """
    # Read data
    minority_data = pd.read_csv(minority_file, header=None).values
    majority_data = pd.read_csv(majority_file, header=None).values

    # Validate feature dimensions
    if minority_data.shape[1] != majority_data.shape[1]:
        raise ValueError(
            f"Feature dimensions don't match: {minority_data.shape[1]} vs {majority_data.shape[1]}"
        )

    # Normalize data
    scaler = MinMaxScaler()
    minority_data = scaler.fit_transform(minority_data)
    majority_data = scaler.transform(majority_data)

    return minority_data, majority_data, scaler


def create_dataloader(data, batch_size=BATCH_SIZE, shuffle=True):
    """
    Create a DataLoader from data.

    Args:
        data (numpy.ndarray): Input data
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the data

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = TabularDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def get_dataset_folders(root_dir):
    """
    Get all dataset folders from the root directory.

    Args:
        root_dir (str): Root directory path

    Returns:
        list: List of dataset folder paths
    """
    return [os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))]


def has_cv_folders(dataset_folder, cv_folds):
    """
    Check if a dataset folder contains all required cross-validation folders.

    Args:
        dataset_folder (str): Path to dataset folder
        cv_folds (list): List of CV fold names

    Returns:
        bool: True if all CV folders exist with required files
    """
    for cv_fold in cv_folds:
        cv_path = os.path.join(dataset_folder, cv_fold)
        if not os.path.isdir(cv_path):
            return False
        if not (os.path.exists(os.path.join(cv_path, 'minority.csv')) and
                os.path.exists(os.path.join(cv_path, 'majority.csv'))):
            return False
    return True


def validate_datasets(data_root_dir, cv_folds):
    """
    Validate and return valid dataset folders.

    Args:
        data_root_dir (str): Root directory for datasets
        cv_folds (list): List of CV fold names

    Returns:
        list: List of valid dataset folder paths
    """
    dataset_folders = get_dataset_folders(data_root_dir)
    valid_dataset_folders = []

    for folder in dataset_folders:
        if has_cv_folders(folder, cv_folds):
            valid_dataset_folders.append(folder)
        else:
            print(f"Warning: {os.path.basename(folder)} does not have proper CV structure, skipping")

    return valid_dataset_folders