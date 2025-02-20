from typing import Iterable
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState
import optax

from nix.mnist.common.networks import sample_z


@struct.dataclass
class Networks:
    encoder: TrainState
    decoder: TrainState
    classifier: TrainState

def binary_cross_entropy_fn(recon, target):
    # Compute binary cross-entropy loss
    recon = jnp.clip(recon, 1e-7, 1 - 1e-7)
    return - (target * jnp.log(recon) + (1 - target) * jnp.log(1 - recon))

def norm_cos_sign_fn(main_grads, aux_grads):
    main_grads_norm = jnp.linalg.norm(main_grads, axis=1, keepdims=True)
    aux_grads_norm = jnp.linalg.norm(aux_grads, axis=1, keepdims=True)

    main_grads_normalized = main_grads / (main_grads_norm+1e-8)
    aux_grads_normalized = aux_grads / (aux_grads_norm+1e-8)

    batch_cos = jnp.sum(main_grads_normalized * aux_grads_normalized, axis=1)

    cos = jnp.mean(batch_cos)
    sign = jnp.mean(jnp.sign(batch_cos))

    return jnp.mean(main_grads_norm), jnp.mean(aux_grads_norm), cos, sign


def train(
    rng: jax.random.PRNGKey,
    networks: Networks,
    train_loader: Iterable,
    kld_coef: float,
):
    @jax.jit
    def train_step(runner_state, batch):
        rng, networks = runner_state
        images, labels = batch

        (mean, logvar), vjp_fn_encoder = jax.vjp(
            lambda params: networks.encoder.apply_fn(params, images),
            networks.encoder.params
        )

        rng, sample_rng = jax.random.split(rng)
        latent_features, vjp_fn_latents = jax.vjp(
            lambda mu, lnv: sample_z(sample_rng, mu, lnv),
            mean, logvar
        )

        # Compute the loss and accuracy for classification
        def classification_loss_fn(params, z):
            class_logits = networks.classifier.apply_fn(params, z)
            classification_loss = jnp.mean(jnp.log(optax.softmax_cross_entropy(class_logits, labels)))

            pred_classes = jnp.argmax(class_logits, axis=-1)
            true_classes = jnp.argmax(labels, axis=-1)
            acc = jnp.mean(jnp.equal(pred_classes, true_classes).astype(jnp.float32))
            return classification_loss, acc

        (classification_loss, acc), (grads_classifier, main_grads) = jax.value_and_grad(classification_loss_fn, argnums=(0, 1), has_aux=True)(networks.classifier.params, latent_features)
        grads_encoder_classification = vjp_fn_encoder(vjp_fn_latents(main_grads))[0] # vjp_fn_latents(main_grads))[0] # (main_grads, jnp.zeros_like(logvar)))[0]

        # Compute the KL divergence between the latent distribution and a standard normal distribution
        def kld_loss_fn(mu, lnv):
            kld_loss = kld_coef * jnp.mean(jnp.log(jnp.sum(-0.5 * (1 + lnv - mu ** 2 - jnp.exp(lnv)), axis=1)))
            return kld_loss

        kld_loss, grads_mean_logvar = jax.value_and_grad(kld_loss_fn, argnums=(0, 1))(mean, logvar)
        grads_encoder_kld = vjp_fn_encoder(grads_mean_logvar)[0]

        # Computes the weighted reconstruction loss
        def recon_loss_fn(params, z):
            recon = networks.decoder.apply_fn(params, z)
            recon_loss_per_pixel = jnp.mean(binary_cross_entropy_fn(recon, images), axis=-1, keepdims=True)
            recon_loss = jnp.mean(jnp.log(jnp.sum(recon_loss_per_pixel, axis=(1, 2, 3))))
            return recon_loss

        recon_loss, (grads_decoder, aux_grads) = jax.value_and_grad(recon_loss_fn, argnums=(0, 1))(networks.decoder.params, latent_features)
        grads_encoder_recon = vjp_fn_encoder(vjp_fn_latents(aux_grads))[0]
        grads_encoder = jax.tree_util.tree_map(lambda x, y, z: x + y + z, grads_encoder_classification, grads_encoder_recon, grads_encoder_kld)

        # Update TrainStates
        new_encoder = networks.encoder.apply_gradients(grads=grads_encoder)
        new_decoder = networks.decoder.apply_gradients(grads=grads_decoder)
        new_classifier = networks.classifier.apply_gradients(grads=grads_classifier)

        # Update networks
        networks = networks.replace(encoder=new_encoder, decoder=new_decoder, classifier=new_classifier)
        main_grads_norm, aux_grads_norm, cos, sign = norm_cos_sign_fn(main_grads, aux_grads)
        metrics = {
            "train/acc": acc,
            "train/classification_loss": classification_loss,
            "train/recon_loss": recon_loss,
            "train/kld_loss": kld_loss,
            "train/main_grads_norm": main_grads_norm,
            "train/aux_grads_norm": aux_grads_norm,
            "train/cos": cos,
            "train/sign": sign,
        }
        return (rng, networks), metrics

    total_metrics = {
        "train/acc": 0.0,
        "train/classification_loss": 0.0,
        "train/recon_loss": 0.0,
        "train/kld_loss": 0.0,
        "train/main_grads_norm": 0.0,
        "train/aux_grads_norm": 0.0,
        "train/cos": 0.0,
        "train/sign": 0.0,
    }

    runner_state = (rng, networks)

    for batch in tqdm(train_loader, leave=False):
        runner_state, metrics = train_step(runner_state, batch)

        for key, value in metrics.items():
            total_metrics[key] += value.item()

    # Compute average metrics
    num_batches = len(train_loader)
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    return runner_state, avg_metrics


def evaluate(
    rng: jax.random.PRNGKey,
    networks: Networks,
    val_loader: Iterable,
    kld_coef: float,
):
    @jax.jit
    def evaluate_step(runner_state, batch):
        rng, networks = runner_state
        images, labels = batch

        mean, logvar = networks.encoder.apply_fn(networks.encoder.params, images)
        latent_features = sample_z(rng, mean, logvar)

        # Computes the loss and accuracy for classification
        def classification_loss_fn(z):
            class_logits = networks.classifier.apply_fn(networks.classifier.params, z)
            classification_loss = jnp.mean(jnp.log(optax.softmax_cross_entropy(class_logits, labels)))

            pred_classes = jnp.argmax(class_logits, axis=-1)
            true_classes = jnp.argmax(labels, axis=-1)
            acc = jnp.mean(jnp.equal(pred_classes, true_classes).astype(jnp.float32))
            return classification_loss, acc

        (classification_loss, acc), main_grads = jax.value_and_grad(classification_loss_fn, has_aux=True)(latent_features)

        # Computes the KL divergence between the latent distribution and a standard normal distribution
        def kld_loss_fn():
            kld_loss = kld_coef * jnp.mean(jnp.log(jnp.sum(-0.5 * (1 + logvar - mean ** 2 - jnp.exp(logvar)), axis=1)))
            return kld_loss

        kld_loss = kld_loss_fn()

        # Computes the weighted reconstruction loss
        def recon_loss_fn(z):
            recon = networks.decoder.apply_fn(networks.decoder.params, z)
            recon_loss_per_pixel = jnp.mean(binary_cross_entropy_fn(recon, images), axis=-1, keepdims=True)
            recon_loss = jnp.mean(jnp.log(jnp.sum(recon_loss_per_pixel, axis=(1, 2))))
            return recon_loss

        recon_loss, aux_grads = jax.value_and_grad(recon_loss_fn)(latent_features)

        main_grads_norm, aux_grads_norm, cos, sign = norm_cos_sign_fn(main_grads, aux_grads)
        metrics = {
            "eval/acc": acc,
            "eval/classification_loss": classification_loss,
            "eval/recon_loss": recon_loss,
            "eval/kld_loss": kld_loss,
            "eval/main_grads_norm": main_grads_norm,
            "eval/aux_grads_norm": aux_grads_norm,
            "eval/cos": cos,
            "eval/sign": sign,
        }
        return metrics

    total_metrics = {
        "eval/acc": 0.0,
        "eval/classification_loss": 0.0,
        "eval/recon_loss": 0.0,
        "eval/kld_loss": 0.0,
        "eval/main_grads_norm": 0.0,
        "eval/aux_grads_norm": 0.0,
        "eval/cos": 0.0,
        "eval/sign": 0.0,
    }

    runner_state = (rng, networks)

    for batch in tqdm(val_loader, leave=False):
        metrics = evaluate_step(runner_state, batch)

        # Accumulate metrics
        for key, value in metrics.items():
            total_metrics[key] += value.item()

    # Compute average metrics
    num_batches = len(val_loader)
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    return avg_metrics
