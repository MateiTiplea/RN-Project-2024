import os
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.mixture import GaussianMixture
from torch import GradScaler, Tensor, nn, optim
from torch.backends import cudnn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
from tqdm import tqdm


# === Device and CUDA setup ===
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_default_device()
cudnn.benchmark = True
pin_memory = True
enable_half = True  # Disable for CPU, it is slower!
scaler = GradScaler(device, enabled=enable_half)


# === Custom Dataset Classes ===
class SimpleCachedDataset(Dataset):
    def __init__(
        self, dataset: Dataset, runtime_transforms: Optional[v2.Transform], cache: bool
    ):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset
        self.runtime_transforms = runtime_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        if self.runtime_transforms is None:
            return image, label
        return self.runtime_transforms(image), label


class CIFAR100_noisy_fine(Dataset):
    """
    See https://github.com/UCSC-REAL/cifar-10-100n, https://www.noisylabels.com/ and `Learning with Noisy Labels
    Revisited: A Study Using Real-World Human Annotations`.
    """

    def __init__(
        self, root: str, train: bool, transform: Optional[Callable], download: bool
    ):
        cifar100 = CIFAR100(
            root=root, train=train, transform=transform, download=download
        )
        data, targets = tuple(zip(*cifar100))

        if train:
            noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(
                    f"{type(self).__name__} need {noisy_label_file} to be used!"
                )

            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError("Clean labels do not match!")

            targets = noise_file["noisy_label"]

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], self.targets[i]


# === Transforms and Dataset Loaders ===
mean = (0.507, 0.4865, 0.4409)
sd = (0.2673, 0.2564, 0.2761)

basic_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

test_transforms = v2.Compose([basic_transforms, v2.Normalize(mean, sd, inplace=True)])

runtime_transforms = v2.Compose(
    [
        v2.RandomCrop(size=32, padding=4),
        v2.RandomHorizontalFlip(0.5),
        v2.Normalize(mean, sd, inplace=True),
    ]
)

# Initialize datasets
train_set = CIFAR100_noisy_fine(
    "./fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100",
    download=False,
    train=True,
    transform=basic_transforms,
)
test_set = CIFAR100_noisy_fine(
    "./fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100",
    download=False,
    train=False,
    transform=test_transforms,
)

# Cache datasets
train_set = SimpleCachedDataset(train_set, runtime_transforms, True)
test_set = SimpleCachedDataset(test_set, None, True)

# Initial data loaders
train_loader = DataLoader(train_set, batch_size=50, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)

# Model initialization
model = timm.create_model("resnext50_32x4d", pretrained=True)
model.fc = nn.Linear(2048, 100)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


# === New functions for noise handling ===
def get_loss_distribution(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    """Calculate the loss distribution for all samples."""
    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device.type, enabled=enable_half):
                outputs = model(inputs)
                # Calculate per-sample cross entropy loss
                loss = torch.nn.functional.cross_entropy(
                    outputs, targets, reduction="none"
                )

                # Check for invalid values
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    # print("Warning: Found NaN or Inf in loss calculation")
                    tqdm.write("Warning: Found NaN or Inf in loss calculation")
                    # Replace invalid values with very high loss
                    loss = torch.nan_to_num(loss, nan=10.0, posinf=10.0, neginf=10.0)

            losses.extend(loss.cpu().tolist())

    # Additional safety check
    losses = np.array(losses)
    if np.isnan(losses).any() or np.isinf(losses).any():
        # print(
        #     "Warning: Found NaN or Inf in final losses, replacing with max valid loss"
        # )
        tqdm.write(
            "Warning: Found NaN or Inf in final losses, replacing with max valid loss"
        )
        valid_max = np.nanmax(losses[np.isfinite(losses)])
        losses = np.nan_to_num(
            losses, nan=valid_max, posinf=valid_max, neginf=valid_max
        )

    return torch.tensor(losses)


def fit_gmm(
    losses: torch.Tensor, n_components: int = 2
) -> Tuple[np.ndarray, GaussianMixture, dict]:
    """
    Fit a Gaussian Mixture Model to identify clean and noisy samples.
    Returns probabilities of samples being clean, the GMM model, and fitting statistics.
    """
    losses_np = losses.numpy().reshape(-1, 1)

    # Remove any remaining invalid values
    valid_mask = np.isfinite(losses_np).ravel()
    if not valid_mask.all():
        # print(
        #     f"Warning: Removing {(~valid_mask).sum()} invalid values before GMM fitting"
        # )
        tqdm.write(
            f"Warning: Removing {(~valid_mask).sum()} invalid values before GMM fitting"
        )
        losses_np = losses_np[valid_mask]

    # Handle empty or single-sample cases
    if len(losses_np) < 2:
        # print("Error: Not enough valid samples for GMM fitting")
        tqdm.write("Error: Not enough valid samples for GMM fitting")
        return (
            np.zeros(len(losses)),
            None,
            {
                "means": np.array([0, 0]),
                "weights": np.array([1, 0]),
                "converged": False,
                "n_iter": 0,
                "lower_bound": 0,
                "clean_ratio": 0,
            },
        )

    # Normalize losses to improve GMM fitting
    losses_mean = np.mean(losses_np)
    losses_std = np.std(losses_np)
    if losses_std == 0:
        # print("Warning: Zero standard deviation in losses")
        tqdm.write("Warning: Zero standard deviation in losses")
        losses_normalized = losses_np - losses_mean
    else:
        losses_normalized = (losses_np - losses_mean) / losses_std

    try:
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=42,
            covariance_type="full",
            max_iter=100,
            n_init=10,
        )
        gmm.fit(losses_normalized)

        # Get probabilities
        probs = gmm.predict_proba(losses_normalized)

        # The component with lower mean is likely the clean component
        clean_component = 0 if gmm.means_[0][0] < gmm.means_[1][0] else 1

        # Calculate fitting statistics
        stats = {
            "means": gmm.means_.flatten() * losses_std + losses_mean,
            "weights": gmm.weights_,
            "converged": gmm.converged_,
            "n_iter": gmm.n_iter_,
            "lower_bound": gmm.lower_bound_,
            "clean_ratio": probs[:, clean_component].mean(),
        }

        # Expand probabilities back to original size if we removed invalid values
        if not valid_mask.all():
            full_probs = np.zeros(len(valid_mask))
            full_probs[valid_mask] = probs[:, clean_component]
            return full_probs, gmm, stats

        return probs[:, clean_component], gmm, stats

    except Exception as e:
        # print(f"Error during GMM fitting: {str(e)}")
        tqdm.write(f"Error during GMM fitting: {str(e)}")
        return (
            np.zeros(len(losses)),
            None,
            {
                "means": np.array([0, 0]),
                "weights": np.array([1, 0]),
                "converged": False,
                "n_iter": 0,
                "lower_bound": 0,
                "clean_ratio": 0,
            },
        )


def split_dataset(
    dataset: Dataset,
    clean_probs: np.ndarray,
    threshold: float = 0.5,
    min_clean_samples: int = 1000,
) -> Tuple[Subset, Subset]:
    """
    Split dataset into clean and noisy subsets based on GMM probabilities.
    Ensures a minimum number of clean samples to prevent batch size issues.

    Args:
        dataset: The dataset to split
        clean_probs: Probabilities of samples being clean
        threshold: Confidence threshold for considering a sample clean
        min_clean_samples: Minimum number of clean samples to maintain
    """
    # Get indices where probability exceeds threshold
    threshold_indices = np.where(clean_probs >= threshold)[0]

    if len(threshold_indices) >= min_clean_samples:
        # If we have enough samples above threshold, use those
        clean_idx = threshold_indices
    else:
        # If not enough samples above threshold, take top-k samples by probability
        sorted_indices = np.argsort(clean_probs)[::-1]
        clean_idx = sorted_indices[:min_clean_samples]
        # print(
        #     f"Warning: Only found {len(threshold_indices)} samples above threshold {threshold}."
        #     f" Taking top {min_clean_samples} samples instead."
        # )
        tqdm.write(
            f"Warning: Only found {len(threshold_indices)} samples above threshold {threshold}."
            f" Taking top {min_clean_samples} samples instead."
        )

    # Get remaining indices for noisy set
    all_indices = set(range(len(dataset)))
    clean_indices_set = set(clean_idx)
    noisy_idx = list(all_indices - clean_indices_set)

    clean_subset = Subset(dataset, clean_idx)
    noisy_subset = Subset(dataset, noisy_idx)

    # print(
    #     f"Split stats: Clean samples: {len(clean_idx)} ({len(clean_idx)/len(dataset)*100:.1f}%), "
    #     f"Noisy samples: {len(noisy_idx)} ({len(noisy_idx)/len(dataset)*100:.1f}%)"
    # )
    tqdm.write(
        f"Split stats: Clean samples: {len(clean_idx)} ({len(clean_idx)/len(dataset)*100:.1f}%), "
        f"Noisy samples: {len(noisy_idx)} ({len(noisy_idx)/len(dataset)*100:.1f}%)"
    )

    return clean_subset, noisy_subset


# === Modified training loop ===
def train_epoch(loader: DataLoader) -> float:
    """Train for one epoch and return accuracy."""
    model.train()
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )

        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


@torch.inference_mode()
def inference():
    model.eval()

    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels


# Main training loop with GMM-based splitting
num_epochs = 50
warmup_epochs = 10  # Increased warmup period to learn better features
update_interval = 3  # Less frequent updates
min_clean_samples = 2000  # Increased minimum clean samples
confidence_threshold = 0.8  # High confidence threshold for clean samples

best_acc = 0.0
epochs = list(range(num_epochs))


def evaluate_gmm_fit(stats: dict) -> bool:
    """Evaluate if GMM fitting is reliable enough to use for splitting."""
    # Check if GMM converged
    if not stats["converged"]:
        return False

    # Check if the components are well-separated
    means = stats["means"]
    if abs(means[0] - means[1]) < 0.5:  # Adjust threshold as needed
        return False

    # Check if clean/noisy ratio makes sense (expecting around 60% clean)
    if not (0.4 <= stats["clean_ratio"] <= 0.8):
        return False

    return True


with tqdm(epochs) as tbar:
    for epoch in tbar:
        if epoch < warmup_epochs:
            # Warm-up training on full dataset
            train_acc = train_epoch(train_loader)
            current_loader = "full"
        else:
            if (epoch - warmup_epochs) % update_interval == 0:
                # Update dataset split using GMM
                losses = get_loss_distribution(model, train_loader)
                clean_probs, gmm, gmm_stats = fit_gmm(losses)

                # Only update split if GMM fitting is reliable
                if evaluate_gmm_fit(gmm_stats):
                    labeled_dataset, unlabeled_dataset = split_dataset(
                        train_set,
                        clean_probs,
                        threshold=confidence_threshold,
                        min_clean_samples=min_clean_samples,
                    )

                    # Adjust batch size based on dataset size
                    batch_size = min(50, max(16, len(labeled_dataset) // 100))

                    # Create new loader for clean samples
                    train_loader = DataLoader(
                        labeled_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=pin_memory,
                        drop_last=True,
                    )
                    current_loader = "clean"

                    # Log GMM statistics
                    # print(f"\nGMM Stats (Epoch {epoch}):")
                    # print(f"Component means: {gmm_stats['means']}")
                    # print(f"Component weights: {gmm_stats['weights']}")
                    # print(f"Clean ratio: {gmm_stats['clean_ratio']:.2f}")
                    # print(f"Clean samples: {len(labeled_dataset)}")
                    tqdm.write(
                        f"\nGMM Stats (Epoch {epoch}):\n"
                        f"Component means: {gmm_stats['means']}\n"
                        f"Component weights: {gmm_stats['weights']}\n"
                        f"Clean ratio: {gmm_stats['clean_ratio']:.2f}\n"
                        f"Clean samples: {len(labeled_dataset)}"
                    )
                else:
                    # print(
                    #     f"\nSkipping GMM split at epoch {epoch} due to unreliable fitting"
                    # )
                    tqdm.write(
                        f"\nSkipping GMM split at epoch {epoch} due to unreliable fitting"
                    )
                    if current_loader == "clean":
                        # Revert to full dataset if current split is unreliable
                        train_loader = DataLoader(
                            train_set,
                            batch_size=50,
                            shuffle=True,
                            pin_memory=pin_memory,
                        )
                        current_loader = "full"

            train_acc = train_epoch(train_loader)

        # Validation
        val_acc = val()
        if val_acc > best_acc:
            best_acc = val_acc

        # Update progress bar
        loader_info = f", {current_loader.capitalize()}"
        if current_loader == "clean":
            loader_info += f": {len(labeled_dataset)}"

        tbar.set_description(
            f"Epoch {epoch}, Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best_acc:.2f}{loader_info}"
        )
