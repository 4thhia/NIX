import hydra
from omegaconf import DictConfig
from tqdm import trange

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint as ocp

from nix.common import Logger
from nix.mnist.common.networks import Classifier, Encoder, Decoder, WeightUNet
from nix.mnist.common.plotter import save_recons_and_weights
from nix.mnist.common.utils import create_ckpt_mngr
from nix.mnist.datasets import create_loaders
from nix.mnist.algos.nix.train import Networks, train, evaluate


def setup_networks(
    rng: jax.random.PRNGKey,
    dummy_inputs: jax.Array,
    num_classes: int,
    zdim: int,
    batch_size: int,
    weight_activation: str,
    lr_encoder: float,
    lr_decoder: float,
    lr_classifier: float,
    lr_weightunet: float,
    lmb: float,
) -> Networks:

    dummy_latents = jnp.zeros((batch_size, zdim))

    # Initialize states with PRNG keys and dummy inputs
    rng_encoder, rng_decoder, rng_classifier, rng_weightunet = jax.random.split(rng, num=4)

    # Setup optimizers
    optim_encoder = optax.adam(lr_encoder)
    optim_decoder = optax.adam(lr_decoder)
    optim_classifier = optax.adam(lr_classifier)
    optim_weightunet = optax.adam(lr_weightunet)

    # Setup models
    encoder = Encoder(color_channels=dummy_inputs.shape[-1], num_latent_features=zdim)
    encoder_params = encoder.init(rng_encoder, dummy_inputs)
    state_encoder = TrainState.create(
        apply_fn=encoder.apply,
        params=encoder_params,
        tx=optim_encoder,
    )

    decoder = Decoder(color_channels=dummy_inputs.shape[-1], decoder_input_size=int(dummy_inputs.shape[1]/2/2))
    decoder_params = decoder.init(rng_decoder, dummy_latents)
    state_decoder = TrainState.create(
        apply_fn=decoder.apply,
        params=decoder_params,
        tx=optim_decoder,
    )

    classifier = Classifier(num_classes=num_classes)
    classifier_params = classifier.init(rng_classifier, dummy_latents)
    state_classifier = TrainState.create(
        apply_fn=classifier.apply,
        params=classifier_params,
        tx=optim_classifier,
    )

    weightunet = WeightUNet(activation=weight_activation)
    weightunet_params = weightunet.init(rng_weightunet, dummy_inputs)
    state_weightunet = TrainState.create(
        apply_fn=weightunet.apply,
        params=weightunet_params,
        tx=optim_weightunet,
    )

    return Networks(state_classifier, state_encoder, state_decoder, state_weightunet, lmb)

@hydra.main(config_path="../_configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run training and evaluation."""
    cfg.run_name = (
        f"{cfg.run_time}__{cfg.algo.name}__{cfg.dataset.name}__{cfg.experiment_name}"
        f"__activation={cfg.weight.activation}__beta={cfg.algo.beta}__lr_lmb={cfg.algo.lmb.lr}"
        f"__kld_coef={cfg.kld_coef}__lr_weightunet={cfg.optimizers.weightunet.lr}__regularization_type={cfg.weight.regularization_type}__regularization_coef={cfg.weight.regularization_coef}"
        f"__intensity={cfg.dataset.intensity}__zdim={cfg.training.zdim}"
    )

    # Setup helpers
    logger = Logger(cfg)
    ckpt_mngr = create_ckpt_mngr(cfg)

    # Setup dataloader
    train_loader, val_loader, ordered_loader, dummy_inputs = create_loaders(cfg.dataset.name, cfg.dataset.intensity, cfg.training.valid_labels, cfg.training.batch_size)

    # Setup model
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, rng_setup = jax.random.split(rng)
    networks = setup_networks(
        rng=rng_setup,
        dummy_inputs=dummy_inputs,
        num_classes=len(cfg.training.valid_labels) ,
        zdim=cfg.training.zdim,
        batch_size=cfg.training.batch_size,
        weight_activation=cfg.weight.activation,
        lr_encoder=cfg.optimizers.encoder.lr,
        lr_decoder=cfg.optimizers.decoder.lr,
        lr_classifier=cfg.optimizers.classifier.lr,
        lr_weightunet=cfg.optimizers.weightunet.lr,
        lmb=cfg.algo.lmb.initial_value,
    )

    best_acc = 0

    for epoch_id in trange(cfg.training.num_epochs, leave=False):

        rng, train_rng, eval_rng = jax.random.split(rng, 3)

        if (epoch_id == 0) or ((epoch_id + 1) % cfg.training.eval_interval == 0):
            metrics = evaluate(
                rng=eval_rng,
                networks=networks,
                val_loader=val_loader,
                kld_coef=cfg.kld_coef,
            )
            logger.log_metrics(metrics)

            best_acc = max(best_acc, metrics['eval/acc'])
            save_recons_and_weights(cfg, f"epoch{epoch_id}", networks, ordered_loader, best_acc)

        (rng, networks), metrics = train(
            rng=train_rng,
            networks=networks,
            train_loader=train_loader,
            kld_coef=cfg.kld_coef,
            weight_regularization_type=cfg.weight.regularization_type,
            weight_regularization_coef=cfg.weight.regularization_coef,
            beta=cfg.algo.beta,
            lr_lmb=cfg.algo.lmb.lr,
        )
        logger.log_metrics(metrics)

        # Save all model states - checkpoint_manager handles interval timing
        states = {
            'encoder': networks.encoder,
            'decoder': networks.decoder,
            'classifier': networks.classifier,
            'weightunet': networks.weightunet
        }
        ckpt_mngr.save(step=epoch_id + 1, args=ocp.args.StandardSave(states))

    save_recons_and_weights(cfg, "final", networks, ordered_loader, best_acc)

if __name__ == "__main__":
    main()