from typing import Tuple, Dict
from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from nix.mnists.common import sample_z


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

@partial(jax.jit, static_argnames=["lr_lmb", "beta", "weight_regularization_type", "weight_regularization_coef"])
def train_step(
    rng: jax.random.PRNGKey,
    state_classifier: TrainState,
    state_encoder: TrainState,
    state_decoder: TrainState,
    state_weightunet: TrainState,
    lmb: float,
    lr_lmb: float,
    beta: float,
    imgs: jax.Array,
    labels: jax.Array,
    weight_regularization_type: str,
    weight_regularization_coef: float,
) -> Tuple[TrainState, TrainState, TrainState, TrainState, float, Dict]:

    (mean, logvar), vjp_fn_encoder = jax.vjp(
        lambda params: state_encoder.apply_fn(params, imgs),
        state_encoder.params
    )

    latent_features, vjp_fn_latents = jax.vjp(
        lambda mu, lnv: sample_z(rng, mu, lnv),
        mean, logvar
    )

    weights, vjp_fn_weightunet = jax.vjp(
        lambda params: state_weightunet.apply_fn(params, imgs),
        state_weightunet.params
    )

    # Computes the loss and accuracy for classification
    def classification_loss_fn(params, z):
        class_logits = state_classifier.apply_fn(state_classifier.params, z)
        classification_loss = jnp.mean(optax.softmax_cross_entropy(class_logits, labels))

        preds = jnp.argmax(class_logits, axis=-1)
        acc = jnp.mean(jnp.equal(preds, jnp.argmax(labels, axis=-1)).astype(jnp.float32))
        return classification_loss, acc

    classification_loss_grads_fn = jax.value_and_grad(classification_loss_fn, argnums=(0, 1), has_aux=True)
    (classification_loss, acc), (grads_classifier, main_grads) = classification_loss_grads_fn(state_classifier.params, latent_features)
    grads_encoder_classification = vjp_fn_encoder(vjp_fn_latents(main_grads))[0]

    # Computes the KL divergence between the latent distribution and a standard normal distribution
    def kld_loss_fn(mu, lnv):
        kld_loss = jnp.mean(jnp.sum(-0.5 * (1 + lnv - mu ** 2 - jnp.exp(lnv)), axis=1))
        return kld_loss

    kld_loss_grads_fn = jax.value_and_grad(kld_loss_fn, argnums=(0, 1))
    kld_loss, grads_mean_logvar = kld_loss_grads_fn(mean, logvar)
    grads_encoder_kld = vjp_fn_encoder(grads_mean_logvar)[0]

    # Computes the weighted reconstruction loss
    def weighted_recon_loss_fn(params, z, ws):
        recon = state_decoder.apply_fn(state_decoder.params, z)
        recon_loss_per_pixel = jnp.mean(binary_cross_entropy_fn(recon, imgs), axis=-1, keepdims=True)
        weighted_recon_loss = jnp.mean(jnp.sum(ws * recon_loss_per_pixel, axis=(1, 2)))
        return weighted_recon_loss

    weighted_recon_loss_grads_fn = jax.value_and_grad(weighted_recon_loss_fn, argnums=0)
    weighted_recon_loss, grads_decoder = weighted_recon_loss_grads_fn(state_decoder.params, latent_features, weights)

    # Computes the weight loss
    def weight_loss_fn(ws):
        aux_grads = jax.grad(weighted_recon_loss_fn, argnums=1)(state_decoder.params, latent_features, ws)
        weight_loss = - lmb * jnp.mean(jnp.sum(main_grads * aux_grads, axis=-1))
        return weight_loss, aux_grads

    weight_loss_grads_fn = jax.value_and_grad(weight_loss_fn, has_aux=True)
    (weight_loss, aux_grads), grads_weights = weight_loss_grads_fn(weights)
    grads_weightunet = vjp_fn_weightunet(grads_weights)[0]

    grads_encoder_recon = vjp_fn_encoder(vjp_fn_latents(aux_grads))[0]
    grads_encoder = jax.tree_util.tree_map(lambda x, y, z: x + y + z, grads_encoder_classification, grads_encoder_recon, grads_encoder_kld)

    # Applies L2 or offset regularization to the weights
    def weight_regularization_loss_fn(ws):
        if weight_regularization_type == "L2":
            weight_regularization_loss = - weight_regularization_coef * jnp.mean(jnp.sum(ws * ws, axis=(1, 2, 3)))
        elif weight_regularization_type == "offset":
            weight_regularization_loss = weight_regularization_coef * jnp.mean(jnp.sum((1 - ws)**2, axis=(1, 2, 3)))
        return weight_regularization_loss

    weight_regularization_loss_grads_fn = jax.value_and_grad(weight_regularization_loss_fn)
    weight_regularization_loss, grads_weights_regularization = weight_regularization_loss_grads_fn(weights)
    grads_weightunet_regularization = vjp_fn_weightunet(grads_weights_regularization)[0]
    grads_weightunet = jax.tree_util.tree_map(lambda x, y: x + y, grads_weightunet, grads_weightunet_regularization)


    # Update TrainStates
    state_classifier = state_classifier.apply_gradients(grads=grads_classifier)
    state_encoder = state_encoder.apply_gradients(grads=grads_encoder)
    state_decoder = state_decoder.apply_gradients(grads=grads_decoder)
    state_weightunet = state_weightunet.apply_gradients(grads=grads_weightunet)
    lmb = jnp.maximum(0.0, lmb + lr_lmb * jnp.mean(jnp.sum(main_grads * (beta * main_grads - aux_grads), axis=1)))


    main_grads_norm, aux_grads_norm, cos, sign = norm_cos_sign_fn(main_grads, aux_grads)
    metrics = {
        "train/acc": acc,
        "train/classification_loss": classification_loss,
        "train/weight_loss": weight_loss,
        "train/weighted_recon_loss": weighted_recon_loss,
        "train/main_grads_norm": main_grads_norm,
        "train/aux_grads_norm": aux_grads_norm,
        "train/cos": cos,
        "train/sign": sign,
        "train/weight_regularization_loss": weight_regularization_loss,
        "train/kld_loss": kld_loss,
    }

    return state_classifier, state_encoder, state_decoder, state_weightunet, lmb, metrics

def train_epoch(
    train_loader,
    rng: jax.random.PRNGKey,
    state_classifier: TrainState,
    state_encoder: TrainState,
    state_decoder: TrainState,
    state_weightunet: TrainState,
    lmb: float,
    lr_lmb: float,
    beta: float,
    weight_regularization_type: str,
    weight_regularization_coef: float,
) -> Tuple[TrainState, TrainState, TrainState, TrainState, float, Dict]:
    total_metrics = {
        "train/acc": 0.0,
        "train/classification_loss": 0.0,
        "train/weight_loss": 0.0,
        "train/weighted_recon_loss": 0.0,
        "train/main_grads_norm": 0.0,
        "train/aux_grads_norm": 0.0,
        "train/cos": 0.0,
        "train/sign": 0.0,
        "train/weight_regularization_loss": 0.0,
        "train/kld_loss": 0.0,
    }
    for imgs, labels in tqdm(train_loader, leave=False):
        rng, sample_rng = jax.random.split(rng)

        state_classifier, state_encoder, state_decoder, state_weightunet, lmb, metrics = train_step(
            sample_rng, state_classifier,
            state_encoder, state_decoder, state_weightunet,
            lmb, lr_lmb, beta,
            imgs, labels,
            weight_regularization_type, weight_regularization_coef,
        )

        # Accumulate metrics
        for key, value in metrics.items():
            total_metrics[key] += value.item()

    # Compute average metrics
    num_batches = len(train_loader)
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    return state_classifier, state_encoder, state_decoder, state_weightunet, lmb, avg_metrics


@partial(jax.jit, static_argnames=["weight_regularization_type", "weight_regularization_coef"])
def eval_step(
    rng: jax.random.PRNGKey,
    state_classifier: TrainState,
    state_encoder: TrainState,
    state_decoder: TrainState,
    state_weightunet: TrainState,
    lmb: float,
    imgs: jax.Array,
    labels: jax.Array,
    weight_regularization_type: str,
    weight_regularization_coef: float,
):

    mean, logvar = state_encoder.apply_fn(state_encoder.params, imgs)

    latent_features = sample_z(rng, mean, logvar)

    weights = state_weightunet.apply_fn(state_weightunet.params, imgs)

    # Computes the loss and accuracy for classification
    def classification_loss_fn(params, z):
        class_logits = state_classifier.apply_fn(state_classifier.params, z)
        classification_loss = jnp.mean(optax.softmax_cross_entropy(class_logits, labels))

        preds = jnp.argmax(class_logits, axis=-1)
        acc = jnp.mean(jnp.equal(preds, jnp.argmax(labels, axis=-1)).astype(jnp.float32))
        return classification_loss, acc

    classification_loss_grads_fn = jax.value_and_grad(classification_loss_fn, argnums=1, has_aux=True)
    (classification_loss, acc), main_grads = classification_loss_grads_fn(state_classifier.params, latent_features)

    # Computes the KL divergence between the latent distribution and a standard normal distribution
    def kld_loss_fn(mu, lnv):
        kld_loss = jnp.mean(jnp.sum(-0.5 * (1 + lnv - mu ** 2 - jnp.exp(lnv)), axis=1))
        return kld_loss

    kld_loss = kld_loss_fn(mean, logvar)

    # Computes the weighted reconstruction loss
    def weighted_recon_loss_fn(params, z, ws):
        recon = state_decoder.apply_fn(state_decoder.params, z)
        recon_loss_per_pixel = jnp.mean(binary_cross_entropy_fn(recon, imgs), axis=-1, keepdims=True)
        weighted_recon_loss = jnp.mean(jnp.sum(ws * recon_loss_per_pixel, axis=(1, 2)))
        return weighted_recon_loss

    weighted_recon_loss = weighted_recon_loss_fn(state_decoder.params, latent_features, weights)

    # Computes the weight loss
    def weight_loss_fn(ws):
        aux_grads = jax.grad(weighted_recon_loss_fn, argnums=1)(state_decoder.params, latent_features, ws)
        weight_loss = - lmb * jnp.mean(jnp.sum(main_grads * aux_grads, axis=-1))
        return weight_loss, aux_grads

    weight_loss, aux_grads = weight_loss_fn(weights)

    # Applies L2 or offset regularization to the weights
    def weight_regularization_loss_fn(ws):
        if weight_regularization_type == "L2":
            weight_regularization_loss = - weight_regularization_coef * jnp.mean(jnp.sum(ws * ws, axis=(1, 2, 3)))
        elif weight_regularization_type == "offset":
            weight_regularization_loss = weight_regularization_coef * jnp.mean(jnp.sum((1 - ws)**2, axis=(1, 2, 3)))
        return weight_regularization_loss

    weight_regularization_loss = weight_regularization_loss_fn(weights)


    # Update TrainStates

    main_grads_norm, aux_grads_norm, cos, sign = norm_cos_sign_fn(main_grads, aux_grads)
    metrics = {
        "eval/acc": acc,
        "eval/classification_loss": classification_loss,
        "eval/weight_loss": weight_loss,
        "eval/weighted_recon_loss": weighted_recon_loss,
        "eval/main_grads_norm": main_grads_norm,
        "eval/aux_grads_norm": aux_grads_norm,
        "eval/cos": cos,
        "eval/sign": sign,
        "eval/weight_regularization_loss": weight_regularization_loss,
        "eval/kld_loss": kld_loss,
    }

    return metrics


def eval_epoch(
    val_loader,
    rng: jax.random.PRNGKey,
    state_classifier: TrainState,
    state_encoder: TrainState,
    state_decoder: TrainState,
    state_weightunet: TrainState,
    lmb: float,
    weight_regularization_type: str,
    weight_regularization_coef: float,
) -> Dict:
    total_metrics = {
        "eval/acc": 0.0,
        "eval/classification_loss": 0.0,
        "eval/weighted_recon_loss": 0.0,
        "eval/weight_loss": 0.0,
        "eval/cos": 0.0,
        "eval/sign": 0.0,
        "eval/main_grads_norm": 0.0,
        "eval/aux_grads_norm": 0.0,
        "eval/kld_loss": 0.0,
        "eval/weight_regularization_loss": 0.0,
    }

    for imgs, labels in tqdm(val_loader, leave=False):
        rng, sample_rng = jax.random.split(rng)

        metrics = eval_step(
            sample_rng, state_classifier,
            state_encoder, state_decoder, state_weightunet, lmb,
            imgs, labels,
            weight_regularization_type, weight_regularization_coef,
        )

        # Accumulate metrics
        for key, value in metrics.items():
            total_metrics[key] += value.item()

    # Compute average metrics
    num_batches = len(val_loader)
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    return avg_metrics