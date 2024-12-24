import hydra
from hydra import utils
from omegaconf import DictConfig, ListConfig
import mlflow
from tqdm import trange


import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint as ocp

from nix.common import Logger, create_ckpt_mngr
from nix.mnists.datasets import create_loaders
from nix.mnists.common import Classifier, Encoder, Decoder, WeightUNet, save_recons_and_weights
from nix.mnists.algos.nix.train import train_epoch, eval_epoch


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


def setup_trainstates(cfg, rng, dummy_inputs):
    dummy_latents = jnp.zeros((cfg.training.batch_size, cfg.training.zdim))

    # Initialize states with PRNG keys and dummy inputs
    rng_classifier, rng_encoder, rng_decoder, rng_weightunet = jax.random.split(rng, num=4)

    # Setup optimizers
    optim_classifier = optax.adam(cfg.optimizers.classifier.lr)
    optim_encoder = optax.adam(cfg.optimizers.encoder.lr)
    optim_decoder = optax.adam(cfg.optimizers.decoder.lr)
    optim_weightunet = optax.adam(cfg.optimizers.weightunet.lr)

    # Setup models
    classifier = Classifier(num_classes=cfg.dataset.num_classes)
    classifier_params = classifier.init(rng_classifier, dummy_latents)
    state_classifier = TrainState.create(
        apply_fn=classifier.apply,
        params=classifier_params,
        tx=optim_classifier,
    )

    encoder = Encoder(color_channels=dummy_inputs.shape[-1], num_latent_features=cfg.training.zdim)
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

    weightunet = WeightUNet(activation=cfg.weight.activation)
    weightunet_params = weightunet.init(rng_weightunet, dummy_inputs)
    state_weightunet = TrainState.create(
        apply_fn=weightunet.apply,
        params=weightunet_params,
        tx=optim_weightunet,
    )

    lmb = cfg.algo.lmb.initial_value

    return state_classifier, state_encoder, state_decoder, state_weightunet, lmb


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run training and evaluation."""
    cfg.run_name = (
        f"{cfg.run_time}__{cfg.experiment_name}__{cfg.sub_experiment_name}"
        f"__activation={cfg.weight.activation}__beta={cfg.algo.beta}"
        f"__lmb={cfg.algo.lmb}__regularization_coef={cfg.weight.regularization_coef}"
        f"__intensity={cfg.dataset.intensity}____zdim={cfg.training.zdim}"
    )

    # Setup helpers
    logger = Logger(cfg)
    ckpt_mngr = create_ckpt_mngr(cfg)

    # Setup dataloader
    train_loader, val_loader, ordered_loader, dummy_inputs = create_loaders(cfg.dataset.name, cfg.dataset.intensity, cfg.training.valid_labels, cfg.training.batch_size)

    # Setup model
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, rng_setup = jax.random.split(rng)
    state_classifier, state_encoder, state_decoder, state_weightunet, lmb = setup_trainstates(cfg, rng_setup, dummy_inputs)

    best_acc = 0

    mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(f"{cfg.algo.name}__lr_w={cfg.optimizers.weightunet.lr}__coef_regu={cfg.weight.regularization_coef}__lr_lmb={cfg.algo.lmb.lr}__{cfg.run_time}")
    with mlflow.start_run():

        for epoch_id in trange(cfg.training.num_epochs, leave=False):

            rng, train_rng, eval_rng = jax.random.split(rng, 3)

            if (epoch_id == 0) or ((epoch_id + 1) % cfg.training.eval_interval == 0):
                metrics = eval_epoch(
                    val_loader,
                    eval_rng, state_classifier,
                    state_encoder, state_decoder, state_weightunet, lmb,
                    cfg.weight.regularization_type,
                    cfg.weight.regularization_coef,
                )
                #logger.log_metrics(metrics)

                best_acc = max(best_acc, metrics['eval/acc'])
                #save_recons_and_weights(cfg, f"epoch{epoch_id}", state_classifier, state_encoder, state_decoder, state_weightunet, ordered_loader, best_acc)
                mlflow.log_metric("best_acc", best_acc, step=(epoch_id+1))

            state_classifier, state_encoder, state_decoder, state_weightunet, lmb, metrics = train_epoch(
                train_loader,
                train_rng, state_classifier,
                state_encoder, state_decoder, state_weightunet,
                lmb, cfg.algo.lmb.lr,
                cfg.algo.beta,
                cfg.weight.regularization_type,
                cfg.weight.regularization_coef,
            )
            #logger.log_metrics(metrics)

            # Save all model states - checkpoint_manager handles interval timing
            states = {
                'classifier': state_classifier,
                'encoder': state_encoder,
                'decoder': state_decoder,
                'weightunet': state_weightunet
            }
            #ckpt_mngr.save(step=epoch_id + 1, args=ocp.args.StandardSave(states))

        #save_recons_and_weights(cfg, "final", state_classifier, state_encoder, state_decoder, state_weightunet, ordered_loader, best_acc)

    return best_acc

if __name__ == "__main__":
    main()