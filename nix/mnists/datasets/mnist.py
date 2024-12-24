import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from nix.mnists.datasets.utils import (
    GrayscaleToRGB,
    create_collate_fn,
    create_ordered_dataset,
    filter_dataset,
)


class MYMNIST(Dataset):
    def __init__(self, intensity: float=1.0, valid_labels: bool=None, root: str='datasets/data', train: bool=True):
        self.intensity = intensity
        # Transforms for MNIST (resize and convert to RGB)
        mnist_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            GrayscaleToRGB(),
            transforms.ToTensor(),
        ])
        # Load MNIST and CIFAR100
        self.mnist_dataset = MNIST(root=root, train=train, download=True, transform=mnist_transform)
        if valid_labels:
            self.mnist_dataset = filter_dataset(self.mnist_dataset, valid_labels)

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, label = self.mnist_dataset[idx]
        combined_img = img * self.intensity

        return combined_img, label #, img #!


def mnist_loader(intensity: float=1.0, valid_labels=None, batch_size: int=64, num_per_class: int=5):
    torch.manual_seed(42)

    # Initialize MNIFAR dataset
    train_dataset = MYMNIST(intensity, valid_labels, train=True)
    test_dataset = MYMNIST(intensity, valid_labels, train=False)

    # Split the training dataset into training and validation sets
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    ordered_dataset = create_ordered_dataset(test_dataset, num_classes=10, num_per_class=num_per_class)

    # Create data loaders for training, validation, and testing
    one_hot_collate = create_collate_fn(num_classes=10)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=one_hot_collate, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=one_hot_collate, shuffle=True, drop_last=True)
    ordered_loader = DataLoader(ordered_dataset, batch_size=(10 * num_per_class), collate_fn=one_hot_collate, shuffle=False)

    # Get a batch of data for model initialization (dummy inputs)
    dummy_inputs = next(iter(train_loader))[0]

    return train_loader, val_loader, ordered_loader, dummy_inputs
