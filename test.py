import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import gc
import warnings
import random
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Hyperparameters
latent_dim = 128


# Generator definition - kept consistent with training code
class Generator(nn.Module):
    def __init__(self, z_dim, data_dim):
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
        return self.main(z)


# Data loading function
def load_and_preprocess_data(minority_file, majority_file):
    minority_data = pd.read_csv(minority_file, header=None).values
    majority_data = pd.read_csv(majority_file, header=None).values

    print(f"Number of minority samples: {len(minority_data)}")
    print(f"Number of majority samples: {len(majority_data)}")

    # Ensure feature dimensions are consistent
    if minority_data.shape[1] != majority_data.shape[1]:
        raise ValueError(f"Error: Minority and majority feature dimensions don't match: {minority_data.shape[1]} vs {majority_data.shape[1]}")

    # Normalize data
    scaler = MinMaxScaler()
    minority_data = scaler.fit_transform(minority_data)
    majority_data = scaler.transform(majority_data)

    return minority_data, majority_data, scaler


# Load trained model
def load_model(model_path, num_generators, data_dim, device):
    generators = []

    try:
        checkpoint = torch.load(model_path, map_location=device)

        for g_idx in range(num_generators):
            generator = Generator(latent_dim, data_dim).to(device)
            generator.load_state_dict(checkpoint['generators_state_dict'][g_idx])
            generator.eval()
            generators.append(generator)

        print(f"Successfully loaded model, current epoch: {checkpoint['epoch']}")

    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return None

    return generators


# Generate samples function - batch processing to reduce memory usage
def generate_samples(generators, num_samples, batch_size, device):
    samples_per_generator = num_samples // len(generators)
    remaining_samples = num_samples % len(generators)

    all_generated_samples = []
    generator_labels = []

    with torch.no_grad():
        for g_idx, generator in enumerate(generators):
            # Calculate how many samples this generator needs to create
            current_samples = samples_per_generator
            if g_idx < remaining_samples:
                current_samples += 1

            generated_samples = []
            remaining = current_samples

            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                z = torch.randn(current_batch_size, latent_dim, device=device)
                fake = generator(z).cpu().numpy()
                generated_samples.extend(fake)
                remaining -= current_batch_size

                # Free memory
                del z, fake
                torch.cuda.empty_cache()

            all_generated_samples.extend(generated_samples)
            generator_labels.extend([g_idx] * len(generated_samples))

            # Free memory
            del generated_samples

    return np.array(all_generated_samples), np.array(generator_labels)


# Visualization function
def visualize_samples(real_data, generated_data, generator_labels, output_path):
    # Combine data for t-SNE
    all_data = np.vstack((real_data, generated_data))

    # Assign different labels for visualization
    # 0: Real minority class, 1+: Samples from different generators
    viz_labels = np.concatenate([
        np.zeros(len(real_data)),  # Real minority class label (0)
        generator_labels + 1  # Generated sample labels (1, 2, 3, ...)
    ])

    # Downsample large datasets to reduce memory usage
    n_samples = len(all_data)
    if n_samples > 1000:
        sample_indices = np.random.choice(n_samples, size=1000, replace=False)
        all_data_sample = all_data[sample_indices]
        viz_labels_sample = viz_labels[sample_indices]
    else:
        all_data_sample = all_data
        viz_labels_sample = viz_labels

    # Adjust perplexity
    perplexity = min(30, len(all_data_sample) - 1)

    try:
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300, random_state=42)
        tsne_results = tsne.fit_transform(all_data_sample)

        # Plot t-SNE visualization
        plt.figure(figsize=(10, 8))

        # Plot real minority class data
        plt.scatter(
            tsne_results[viz_labels_sample == 0, 0],
            tsne_results[viz_labels_sample == 0, 1],
            c='blue', label='Real Minority', alpha=0.7, s=40
        )

        # Plot samples generated by each generator
        colors = ['green', 'purple', 'orange', 'cyan', 'magenta']
        num_generators = len(np.unique(generator_labels))

        for g_idx in range(num_generators):
            mask = viz_labels_sample == g_idx + 1
            if np.any(mask):  # Ensure there are samples to plot
                plt.scatter(
                    tsne_results[mask, 0],
                    tsne_results[mask, 1],
                    c=colors[g_idx % len(colors)],
                    label=f'Generator {g_idx}',
                    alpha=0.5, s=40
                )

        plt.title('t-SNE Visualization (Minority & Generated Samples)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        # Free memory
        del tsne_results, all_data_sample, viz_labels_sample

    except Exception as e:
        print(f"t-SNE visualization failed: {str(e)}")

    # Free memory
    del all_data, viz_labels
    gc.collect()


# Create balanced dataset
def create_balanced_dataset(minority_data, majority_data, generated_data, scaler, output_path):
    # Calculate how many samples are needed
    samples_needed = len(majority_data) - len(minority_data)

    # Use the first samples_needed samples from generated data
    if len(generated_data) >= samples_needed:
        balanced_minority_data = np.vstack((minority_data, generated_data[:samples_needed]))
    else:
        print(f"Warning: Not enough generated samples, need {samples_needed} but only have {len(generated_data)}")
        # If not enough samples, repeat the generated samples
        repeat_times = samples_needed // len(generated_data) + 1
        repeated_data = np.tile(generated_data, (repeat_times, 1))
        balanced_minority_data = np.vstack((minority_data, repeated_data[:samples_needed]))

    # Create labels
    majority_labels = np.zeros(len(majority_data))
    minority_labels = np.ones(len(balanced_minority_data))

    # Combine features and labels
    X = np.vstack((majority_data, balanced_minority_data))
    y = np.concatenate((majority_labels, minority_labels))

    # Inverse transform data (if needed)
    if scaler is not None:
        X = scaler.inverse_transform(X)

    # Create complete dataset
    data_with_labels = np.column_stack((X, y))

    # Shuffle data randomly
    np.random.shuffle(data_with_labels)

    # Save balanced dataset
    df = pd.DataFrame(data_with_labels)
    df.to_csv(output_path, index=False, header=False)

    print(f"Balanced dataset saved to: {output_path}")
    print(f"Balanced dataset shape: {data_with_labels.shape}")

    # Verify class balance
    labels_count = np.bincount(data_with_labels[:, -1].astype(int))
    print(f"Class counts: Class 0 (majority): {labels_count[0]}, Class 1 (minority+generated): {labels_count[1]}")


# Find the latest model file
def find_latest_model(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.startswith('models_') and f.endswith('.pth')]
    if not model_files:
        return None

    # Extract epoch number and sort
    model_epochs = [(f, int(f.split('_')[1].split('.')[0])) for f in model_files]
    latest_model = sorted(model_epochs, key=lambda x: x[1], reverse=True)[0][0]

    return os.path.join(model_dir, latest_model)


# Main function - Modified for 5-fold cross-validation structure
def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Get dataset folders (top-level folders in data_root_dir)
    def get_dataset_folders(root_dir):
        return [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))]

    # Check if a folder contains cross-validation folders
    def has_cv_folders(dataset_folder):
        for cv_fold in ['1cv', '2cv', '3cv', '4cv', '5cv']:
            cv_path = os.path.join(dataset_folder, cv_fold)
            if not os.path.isdir(cv_path):
                return False
            if not (os.path.exists(os.path.join(cv_path, 'minority.csv')) and
                    os.path.exists(os.path.join(cv_path, 'majority.csv'))):
                return False
        return True

    # Set data folder and output folder - matching the training structure
    data_root_dir = './data'  # Same as training code
    model_root_dir = './generated/MDGAN-CV'  # From training code
    test_output_dir = './generated/MDGAN-Test'  # Output directory for test results

    # Ensure output directory exists
    os.makedirs(test_output_dir, exist_ok=True)

    # Get all dataset folders
    dataset_folders = get_dataset_folders(data_root_dir)
    valid_dataset_folders = []

    # Validate dataset folders (should contain 5 CV folders)
    for folder in dataset_folders:
        if has_cv_folders(folder):
            valid_dataset_folders.append(folder)
        else:
            print(f"Warning: {os.path.basename(folder)} does not have proper CV structure, skipping")

    # Check if there are valid dataset folders
    if not valid_dataset_folders:
        print(f"No valid datasets with 5-fold CV structure found in: {data_root_dir}")
        return

    print(f"Found {len(valid_dataset_folders)} valid datasets with 5-fold CV structure")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Batch processing size
    batch_size = 32
    # Number of generators (matching the training code)
    num_generators = 3

    # Process each dataset with its 5-fold CV structure
    for dataset_folder in tqdm(valid_dataset_folders, desc="Processing datasets"):
        dataset_name = os.path.basename(dataset_folder)
        print(f"\nProcessing dataset: {dataset_name}")

        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(test_output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Process each CV fold
        for cv_fold in ['1cv', '2cv', '3cv', '4cv', '5cv']:
            print(f"\nProcessing fold: {cv_fold}")

            # Paths for current fold
            cv_fold_path = os.path.join(dataset_folder, cv_fold)
            minority_file = os.path.join(cv_fold_path, 'minority.csv')
            majority_file = os.path.join(cv_fold_path, 'majority.csv')

            # Create fold-specific output directory
            fold_output_dir = os.path.join(dataset_output_dir, cv_fold)
            os.makedirs(fold_output_dir, exist_ok=True)

            # Model directory (from training output)
            model_dir = os.path.join(model_root_dir, dataset_name, cv_fold)

            # Check if model directory exists
            if not os.path.exists(model_dir):
                print(f"Model directory does not exist: {model_dir}, skipping this fold")
                continue

            # Find the latest model file
            model_path = find_latest_model(model_dir)
            if model_path is None:
                print(f"No model files found in {model_dir}, skipping this fold")
                continue

            print(f"Using model: {model_path}")

            try:
                # Load and preprocess data
                minority_data, majority_data, scaler = load_and_preprocess_data(minority_file, majority_file)

                # Get data dimension
                data_dim = minority_data.shape[1]

                # Load model
                generators = load_model(model_path, num_generators, data_dim, device)
                if generators is None:
                    print(f"Unable to load model, skipping this fold")
                    continue

                print("Generating samples...")

                # Part 1: Generate samples equal to minority class count for visualization
                num_viz_samples = len(minority_data)
                viz_samples, viz_generator_labels = generate_samples(generators, num_viz_samples, batch_size, device)

                # Save visualization samples
                viz_samples_path = os.path.join(fold_output_dir, 'visualization_samples.csv')
                pd.DataFrame(viz_samples).to_csv(viz_samples_path, index=False, header=False)
                print(f"Visualization samples saved to: {viz_samples_path}")

                # Visualize the distribution of generated vs. real minority samples
                viz_output_path = os.path.join(fold_output_dir, 'tsne_visualization.png')
                visualize_samples(minority_data, viz_samples, viz_generator_labels, viz_output_path)
                print(f"Visualization saved to: {viz_output_path}")

                # Part 2: Generate samples for balancing
                samples_needed = len(majority_data) - len(minority_data)
                balance_samples, _ = generate_samples(generators, samples_needed, batch_size, device)

                # Create and save balanced dataset
                balanced_dataset_path = os.path.join(fold_output_dir, 'balanced_dataset.csv')
                create_balanced_dataset(minority_data, majority_data, balance_samples, scaler, balanced_dataset_path)

                # Free memory
                del minority_data, majority_data, viz_samples, balance_samples, generators
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"Error processing {dataset_name}/{cv_fold}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    print("All datasets testing completed!")


if __name__ == "__main__":
    main()
