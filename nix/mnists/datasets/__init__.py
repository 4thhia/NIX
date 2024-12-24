from nix.mnists.datasets.mnist import mnist_loader
from nix.mnists.datasets.mnifar import mnifar_loader
from nix.mnists.datasets.emnist import emnist_loader
from nix.mnists.datasets.emnifar import emnifar_loader

def create_loaders(dataset: str, intensity: int=1.0, valid_labels=None, batch_size: int=64):
    if dataset == "mnist":
        return mnist_loader(intensity, valid_labels, batch_size)
    elif dataset == "mnifar":
        return mnifar_loader(intensity, valid_labels, batch_size)
    elif dataset == "emnist":
        return emnist_loader(intensity, valid_labels, batch_size)
    elif dataset == "emnifar":
        return emnifar_loader(intensity, valid_labels, batch_size)