import os
from typing import Tuple
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import numpy as np
import jax
import jax.numpy as jnp

from nix.mnist.common.networks import sample_z


def label_to_alphabet(labels: np.ndarray) -> np.ndarray:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    indices = np.argmax(labels, axis=-1)
    return np.array([alphabet[index] for index in indices])

@jax.jit
def create_recons(
    networks,
    images: jnp.ndarray,
) -> jax.Array:
    rng = jax.random.PRNGKey(42)
    mean, logvar = networks.encoder.apply_fn(networks.encoder.params, images)
    latent_features = sample_z(rng, mean, logvar)
    recons = networks.decoder.apply_fn(networks.decoder.params, latent_features)
    return recons

@jax.jit
def create_recons_and_weights(
    networks,
    images: jnp.ndarray,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    rng = jax.random.PRNGKey(42)
    mean, logvar = networks.encoder.apply_fn(networks.encoder.params, images)
    latent_features = sample_z(rng, mean, logvar)

    # Compute preds
    class_logits = networks.classifier.apply_fn(networks.classifier.params, latent_features)

    # Compute recons
    recons = networks.decoder.apply_fn(networks.decoder.params, latent_features)

    # Compute weights
    weights = networks.weightunet.apply_fn(networks.weightunet.params, images)

    return class_logits, recons, weights


def plot_image(cfg, prefix, header, data):
    """
    Save the image to a specified location.

    Args:
        cfg: A dictionary containing configuration settings. This may include keys such as:
            - 'experiment_name' (str): The name of the image file (excluding the file extension).
            - 'sub_experiment_name' (str): The timestamp when the image was generated or saved.
            - 'weight.normalizer' (str, optional): The name of the person who generated or saved the image.
            - 'run_name' (str, optional): A short description or context about the image.

        prefix: initial / middle / final

        headers (dict): A dictionary containing metadata related to the image. This may include keys such as:
            - 'normalizer': const or sigmoid or tanh. #! weight.normalizer
            - 'loss_type': ninnerproduct or l2. #! weight.loss_type
            - 'zdim': 2 or 16. #! training.zdim
            - 'acc': float. #! best_acc

        data list: A list of dict containing image data and associated details. This may include keys such as:
            - 'image_title': .
            - 'image': .
            - 'plot_theme': gray or rgb or warm or coolwarm

    Returns:
        None
    """
    img_dir = f"out/{cfg.algo.name}/{cfg.dataset.name}/{cfg.weight.regularization_type}/{cfg.experiment_name}/{cfg.run_time}/"

    os.makedirs(img_dir, exist_ok=True)
    save_path = img_dir + prefix + f"__{cfg.run_name}.pdf"

    # custom colormap: coolwarm.white -> coolwarm.red
    coolwarm = plt.get_cmap("coolwarm")
    colors = coolwarm(np.linspace(0.5, 1, 256))
    custom_cmap = LinearSegmentedColormap.from_list("custom_white_red", colors)

    NUM_COLUMNS = len(data)
    NUM_ROWS = len(data[0]["datum"])
    fig, axes = plt.subplots(NUM_ROWS, NUM_COLUMNS, figsize=(NUM_COLUMNS*5, NUM_ROWS*2))

    for i in range(NUM_ROWS):
        for j in range(NUM_COLUMNS):
            # Display the first image
            if data[j]["plot_theme"] == "gray":
                axes[i, j].imshow(data[j]["datum"][i], cmap="gray")
            elif data[j]["plot_theme"] == "rgb":
                axes[i, j].imshow(data[j]["datum"][i])
            elif data[j]["plot_theme"] == "coolwarm":
                sns.heatmap(data[j]["datum"][i], cmap="coolwarm", vmin=-1, vmax=1, cbar=False, square=True, ax=axes[i, j], xticklabels=False, yticklabels=False)
            elif data[j]["plot_theme"] == "warm":
                sns.heatmap(data[j]["datum"][i], cmap=custom_cmap, cbar=False, square=True, ax=axes[i, j], xticklabels=False, yticklabels=False)

            # Set subtitle
            if isinstance(data[j]["image_title"], dict):
                key, value = list(data[j]["image_title"].items())[0]
                if isinstance(value, (list, np.ndarray, jnp.ndarray)):
                    axes[i, j].set_title(f'{key}:{value[i]}', fontsize=12)
                else:
                    axes[i, j].set_title(f'{key}:{value}', fontsize=12)
            else:
                axes[i, j].set_title(data[j]["image_title"], fontsize=12)
            axes[i, j].axis("off")

            if j > 0:
                pos_img = axes[i, j-1].get_position()
                axes[i, j].set_position([pos_img.x1, pos_img.y0, pos_img.width, pos_img.height])

    title = "\n".join([f"{key}:{value}" for key, value in header.items()])
    plt.suptitle(title, fontsize=16, x=0.33, y=0.9175)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved output to {save_path}")


def save_recons(cfg, prefix, networks, ordered_loader, best_acc):
    images, labels = next(iter(ordered_loader))
    recons = np.asarray(create_recons(networks, images))

    if cfg.dataset.name in {"mnist", "mnifar"}:
        labels = np.asarray(np.argmax(labels, axis=-1))
    elif cfg.dataset.name in {"emnist", "emnifar"}:
        labels = label_to_alphabet(labels)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}. Expected mnifar or emnifar.")


    header = {
        "algo": cfg.algo.name,
        "zdim": cfg.training.zdim,
        "run_time": cfg.run_time,
        "acc": best_acc if best_acc else None,
    }

    data = [
        {"image_title": {"labels": labels}, "datum": images, "plot_theme": "rgb"},
        {"image_title": {"labels": labels}, "datum": recons, "plot_theme": "rgb"},
    ]

    plot_image(cfg, prefix, header, data)


def save_recons_and_weights(cfg, prefix, networks, ordered_loader, best_acc=None):
    images, labels = next(iter(ordered_loader))

    class_logits, recons, weights = create_recons_and_weights(networks, images)

    if cfg.dataset.name in {"mnist", "mnistfashion", "mnifar"}:
        labels = np.asarray(np.argmax(labels, axis=-1))
        preds = np.argmax(class_logits, axis=-1)
    elif cfg.dataset.name in {"emnist", "emnistfashion", "emnifar"}:
        labels = label_to_alphabet(labels)
        preds = label_to_alphabet(class_logits)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}. Expected mnifar or emnifar.")

    recons, weights = np.asarray(recons), np.asarray(weights.squeeze())

    header = {
        "algo": cfg.algo.name,
        "regularizer": f"{cfg.weight.regularization_type}:{cfg.weight.regularization_coef}",
        "lr_wightunet": f"{cfg.optimizers.weightunet.lr}",
        "activation": cfg.weight.activation,
        "zdim": cfg.training.zdim,
        "run_time": cfg.run_time,
        "acc": best_acc if best_acc else None,
    }

    data = [
        {"image_title": {"labels": labels}, "datum": images, "plot_theme": "rgb"},
        {"image_title": {"preds": preds}, "datum": recons, "plot_theme": "rgb"},
        {"image_title": "weight", "datum": weights, "plot_theme": "coolwarm" if cfg.weight.activation == 'tanh' else "warm"}
    ]

    plot_image(cfg, prefix, header, data)