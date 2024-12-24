import random

import numpy as np
from torch.utils.data import Subset

class GrayscaleToRGB:
    def __call__(self, img):
        # Convert grayscale image to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img


def create_collate_fn(num_classes=10, one_hot=True):
    def flax_collate(batch):
        combined_imgs, labels = zip(*batch)
        #combined_imgs, labels, trueweights = zip(*batch)

        # Convert to NumPy arrays and transpose (0, 2, 3, 1)
        combined_imgs = np.stack([img.numpy().transpose(1, 2, 0) for img in combined_imgs])
        #trueweights = np.stack([weight.numpy().transpose(1, 2, 0) for weight in trueweights])

        if one_hot:
            # Convert labels to one-hot encoding
            labels = np.array([np.eye(num_classes)[label] for label in labels])

        return combined_imgs, labels #, trueweights
    return flax_collate


def create_ordered_dataset(dataset, num_classes, num_per_class):

    if len(dataset) < num_classes * num_per_class:
        print(f"Size of {dataset.__class__.__name__} is smaller than num_classes * num_per_class: {num_classes * num_per_class}")
        return dataset

    indices_per_class = {i: [] for i in range(num_classes)}
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    for idx in all_indices:
        _, label = dataset[idx]
        if  len(indices_per_class[label]) < num_per_class:
            indices_per_class[label].append(idx)
        if all(len(indices) == num_per_class for indices in indices_per_class.values()):
            break

    # Create an ordered list of indices (000001111122222...)
    ordered_indices = [idx for i in range(num_classes) for idx in indices_per_class[i]]
    ordered_dataset = Subset(dataset, ordered_indices)

    return ordered_dataset


def filter_dataset(dataset, valid_labels):
    indices = [i for i, label in enumerate(dataset.targets) if label in valid_labels]
    filtered_data = Subset(dataset, indices)
    return filtered_data
