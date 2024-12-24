import torch
from torchvision.datasets import EMNIST, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from nix.mnists.datasets.utils import (
    GrayscaleToRGB,
    create_collate_fn,
    create_ordered_dataset,
    filter_dataset,
)

class EMNIFAR(Dataset):
    def __init__(self, intensity: float=1.0, valid_labels: bool=None, root: str='datasets/data', train: bool=True):
        self.intensity = intensity
        # Transforms for EMNIST (resize and convert to RGB)
        emnist_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            GrayscaleToRGB(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
        ])
        # Transform for CIFAR100 (just convert to tensor)
        cifar100_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # Load MNIST and CIFAR100
        emnist_dataset = EMNIST(root=root, split='letters', train=train, download=True, transform=emnist_transform)
        if valid_labels:
            emnist_dataset = filter_dataset(emnist_dataset, valid_labels)

        self.cifar100_dataset = CIFAR100(root=root, train=True, download=True, transform=cifar100_transform)
        self.emnist_dataset = Subset(emnist_dataset, list(range(min(len(emnist_dataset), len(self.cifar100_dataset)))))

    def __len__(self):
        return len(self.emnist_dataset)

    def __getitem__(self, idx):
        img, label = self.emnist_dataset[idx]
        background_img, _ = self.cifar100_dataset[idx]

        # Create mask based on MNIST pixel intensity (> 0.4)
        mask = (img[0] > 0.4).unsqueeze(0).float()
        combined_img = mask * img * self.intensity + (1 - mask) * background_img

        # Create ideal weight
        lmask = torch.cat((mask[:, :, 4:], mask[:, :, :4]), dim=2)
        rmask = torch.cat((mask[:, :, -4:], mask[:, :, :-4]), dim=2)
        bmask = torch.cat((mask[:, -4:, :], mask[:, :-4, :]), dim=1)
        tmask = torch.cat((mask[:, 4:, :], mask[:, :4, :]), dim=1)
        trueweight = torch.logical_or(torch.logical_or(torch.logical_or(lmask, rmask), bmask), tmask).float()

        return combined_img, label - 1 #, trueweight #!


def emnifar_loader(intensity: float=1.0, valid_labels=None, batch_size: int=64, num_per_class: int=5):
    torch.manual_seed(42)

    # Initialize EMNIFAR dataset
    train_dataset = EMNIFAR(intensity, valid_labels, train=True)
    test_dataset = EMNIFAR(intensity, valid_labels, train=False)

    # Split the training dataset into training and validation sets
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    ordered_dataset = create_ordered_dataset(test_dataset, num_classes=26, num_per_class=num_per_class)

    # Create data loaders for training, validation, and testing
    one_hot_collate = create_collate_fn(num_classes=26)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=one_hot_collate, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=one_hot_collate, shuffle=True, drop_last=True)
    ordered_loader = DataLoader(ordered_dataset, batch_size=(26 * num_per_class), collate_fn=one_hot_collate, shuffle=False)

    # Get a batch of data for model initialization (dummy inputs)
    dummy_inputs = next(iter(train_loader))[0]

    return train_loader, val_loader, ordered_loader, dummy_inputs