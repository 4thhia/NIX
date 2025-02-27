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
    weightunet: TrainState
    lmb: float
    gamma_coef: float


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
    weight_regularization_type: str,
    weight_regularization_coef: float,
    beta: float,
    lr_lmb: float,
    gamma_max: float,
    gamma_coef_bound: float,
    lr_gamma_coef: float,
    target_loss: float,
):
    # Applies negative_square or offset regularization to the weights
    def weight_regularization_loss_fn(weights):
        if weight_regularization_type == "negative_square":
            weight_regularization_loss = - weight_regularization_coef * jnp.mean(jnp.sum(weights * weights, axis=(1, 2, 3)))
        elif weight_regularization_type == "offset":
            weight_regularization_loss = weight_regularization_coef * jnp.mean(jnp.sum((1 - weights)**2, axis=(1, 2, 3)))
        elif weight_regularization_type == "smooth":
            diff_x = weights[:, :, :-1, :] - weights[:, :, 1:, :]
            diff_y = weights[:, :-1, :, :] - weights[:, 1:, :, :]
            weight_regularization_loss = weight_regularization_coef * jnp.mean(jnp.sum(diff_x**2, axis=(1, 2, 3))) + jnp.mean(jnp.sum(diff_y**2, axis=(1, 2, 3)))
        else:
            weight_regularization_loss = 0
        return weight_regularization_loss

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

        weights, vjp_fn_weightunet = jax.vjp(
            lambda params: networks.weightunet.apply_fn(params, images),
            networks.weightunet.params
        )

        # Computes the loss and accuracy for classification
        def classification_loss_fn(params, mu):
            class_logits = networks.classifier.apply_fn(params, mu)
            classification_loss = jnp.mean(optax.softmax_cross_entropy(class_logits, labels))

            pred_classes = jnp.argmax(class_logits, axis=-1)
            true_classes = jnp.argmax(labels, axis=-1)
            acc = jnp.mean(jnp.equal(pred_classes, true_classes).astype(jnp.float32))
            return classification_loss, acc

        (classification_loss, acc), (grads_classifier, main_grads) = jax.value_and_grad(classification_loss_fn, argnums=(0, 1), has_aux=True)(networks.classifier.params, latent_features)
        grads_encoder_classification = vjp_fn_encoder((main_grads, jnp.zeros_like(logvar)))[0]

        # Computes the KL divergence between the latent distribution and a standard normal distribution
        def kld_loss_fn(mu, lnv):
            kld_loss = jnp.mean(jnp.sum(-0.5 * (1 + lnv - mu ** 2 - jnp.exp(lnv)), axis=1))
            return kld_loss

        kld_loss, grads_mean_logvar = jax.value_and_grad(kld_loss_fn, argnums=(0, 1))(mean, logvar)
        grads_encoder_kld = vjp_fn_encoder(grads_mean_logvar)[0]

        # Computes the weighted reconstruction loss
        def weighted_recon_loss_fn(params, z, ws):
            recon = networks.decoder.apply_fn(params, z)
            recon_loss_per_pixel = jnp.mean(binary_cross_entropy_fn(recon, images), axis=-1, keepdims=True)
            weighted_recon_loss = jnp.mean(jnp.sum(ws * recon_loss_per_pixel, axis=(1, 2)))
            return weighted_recon_loss

        weighted_recon_loss, grads_decoder = jax.value_and_grad(weighted_recon_loss_fn, argnums=0)(networks.decoder.params, latent_features, weights)

        # Computes the weight loss
        def weight_loss_fn(ws):
            aux_grads = jax.grad(weighted_recon_loss_fn, argnums=1)(networks.decoder.params, latent_features, ws)
            weight_loss = - networks.lmb * jnp.mean(jnp.sum(main_grads * aux_grads, axis=-1))
            return weight_loss, aux_grads

        (weight_loss, aux_grads), grads_weights_main = jax.value_and_grad(weight_loss_fn, has_aux=True)(weights)
        grads_weightunet_main = vjp_fn_weightunet(grads_weights_main)[0]

        grads_encoder_recon = vjp_fn_encoder(vjp_fn_latents(aux_grads))[0]
        grads_encoder = jax.tree_util.tree_map(lambda x, y, z: x + y + z, grads_encoder_classification, grads_encoder_recon, grads_encoder_kld)

        # Computes the weight regularization loss
        weight_regularization_loss, grads_weights_regulalrization = jax.value_and_grad(weight_regularization_loss_fn)(weights)
        grads_weightunet_regularization = vjp_fn_weightunet(grads_weights_regulalrization)[0]
        grads_weightunet = jax.tree_util.tree_map(lambda x, y: x + y, grads_weightunet_main, grads_weightunet_regularization)

        # Update TrainStates
        new_encoder = networks.encoder.apply_gradients(grads=grads_encoder)
        new_decoder = networks.decoder.apply_gradients(grads=grads_decoder)
        new_classifier = networks.classifier.apply_gradients(grads=grads_classifier)
        new_weightunet = networks.weightunet.apply_gradients(grads=grads_weightunet)

        # Update gamma
        new_gamma_coef = jnp.clip(networks.gamma_coef + lr_gamma_coef * (target_loss - classification_loss), -gamma_coef_bound, gamma_coef_bound)
        gamma = (target_loss > classification_loss) * nn.sigmoid(new_gamma_coef) * gamma_max

        # Update lambda
        new_lmb = jnp.maximum(0.0, networks.lmb + lr_lmb * (jnp.mean(jnp.sum(main_grads * (beta * main_grads - aux_grads), axis=1)) - gamma))

        networks = networks.replace(encoder=new_encoder, decoder=new_decoder, classifier=new_classifier, weightunet=new_weightunet, lmb=new_lmb, gamma_coef=new_gamma_coef)

        main_grads_norm, aux_grads_norm, cos, sign = norm_cos_sign_fn(main_grads, aux_grads)
        metrics = {
            "train/acc": acc,
            "train/classification_loss": classification_loss,
            "train/weight_loss": weight_loss,
            "train/weighted_recon_loss": weighted_recon_loss,
            "train/weight_regularization_loss": weight_regularization_loss,
            "train/kld_loss": kld_loss,
            "train/lmb": new_lmb,
            "train/gamma_coef": new_gamma_coef,
            "train/main_grads_norm": main_grads_norm,
            "train/aux_grads_norm": aux_grads_norm,
            "train/cos": cos,
            "train/sign": sign,
        }
        return (rng, networks), metrics

    total_metrics = {
        "train/acc": 0.0,
        "train/classification_loss": 0.0,
        "train/weight_loss": 0.0,
        "train/weighted_recon_loss": 0.0,
        "train/weight_regularization_loss": 0.0,
        "train/kld_loss": 0.0,
        "train/lmb": 0.0,
        "train/gamma_coef": 0.0,
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
    def evaluate_step(runner_state, batch):
        rng, networks = runner_state
        images, labels = batch

        mean, logvar = networks.encoder.apply_fn(networks.encoder.params, images)
        latent_features = sample_z(rng, mean, logvar)
        weights = networks.weightunet.apply_fn(networks.weightunet.params, images)

        # Computes the loss and accuracy for classification
        def classification_loss_fn(z):
            class_logits = networks.classifier.apply_fn(networks.classifier.params, z)
            classification_loss = jnp.mean(optax.softmax_cross_entropy(class_logits, labels))

            preds = jnp.argmax(class_logits, axis=-1)
            acc = jnp.mean(jnp.equal(preds, jnp.argmax(labels, axis=-1)).astype(jnp.float32))
            return classification_loss, acc

        classification_loss_grads_fn = jax.value_and_grad(classification_loss_fn, has_aux=True)
        (classification_loss, acc), main_grads = classification_loss_grads_fn(latent_features)

        # Computes the KL divergence between the latent distribution and a standard normal distribution
        def kld_loss_fn():
            kld_loss = jnp.mean(jnp.sum(-0.5 * (1 + logvar - mean ** 2 - jnp.exp(logvar)), axis=1))
            return kld_loss

        kld_loss = kld_loss_fn()

        # Computes the weighted reconstruction loss
        def weighted_recon_loss_fn(z):
            recon = networks.decoder.apply_fn(networks.decoder.params, z)
            recon_loss_per_pixel = jnp.mean(binary_cross_entropy_fn(recon, images), axis=-1, keepdims=True)
            weighted_recon_loss = jnp.mean(jnp.sum(weights * recon_loss_per_pixel, axis=(1, 2)))
            return weighted_recon_loss

        weighted_recon_loss, aux_grads = jax.value_and_grad(weighted_recon_loss_fn)(latent_features)

        main_grads_norm, aux_grads_norm, cos, sign = norm_cos_sign_fn(main_grads, aux_grads)
        metrics = {
            "eval/acc": acc,
            "eval/classification_loss": classification_loss,
            "eval/weighted_recon_loss": weighted_recon_loss,
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
        "eval/weighted_recon_loss": 0.0,
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
