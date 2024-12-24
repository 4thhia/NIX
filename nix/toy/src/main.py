import os
import json
import hydra
from omegaconf import DictConfig
from tqdm import trange

import numpy as np
import jax
import jax.numpy as jnp

from gits.toy.src.utils import plot_lambda, plot2D, plot3D
from gits.toy.src.losses import create_loss_grad_fn


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run training and evaluation."""
    os.makedirs(f"out/{cfg.experiment_name}/{cfg.sub_experiment_name}", exist_ok=True)

    _, main_grad_fns = create_loss_grad_fn(cfg.main.mux, cfg.main.muy, cfg.main.stdx, cfg.main.stdy, cfg.main.rho, cfg.main.flat)
    _, aux_grad_fns = create_loss_grad_fn(cfg.aux.mux, cfg.aux.muy, cfg.aux.stdx, cfg.aux.stdy, cfg.aux.rho, cfg.aux.flat)

    @jax.jit
    def update_step(params, weights, lmb):
        main_grad = jnp.sum(jnp.array([fn(params) for fn in main_grad_fns]), axis=0)
        _main_grad = jnp.copy(main_grad)
        if cfg.training.normalize == True:
            _main_grad = _main_grad / jnp.linalg.norm(_main_grad)
        aux_grads = [fn(params) for fn in aux_grad_fns]

        def loss_fn(weights):
            weighted_aux_grad = jnp.sum(jnp.array([w * g for w, g in zip(weights, aux_grads)]), axis=0)
            weight_loss = - jnp.sum(_main_grad * weighted_aux_grad)
            return weight_loss, weighted_aux_grad

        (_, weighted_aux_grad), weight_grad = jax.value_and_grad(loss_fn, has_aux=True)(weights)

        next_params = params - cfg.optimizers.lr_params * (main_grad + weighted_aux_grad)
        if cfg.training.regularizer == "squared_loss":
            next_weights = weights - cfg.optimizers.lr_weights * (weight_grad - 0.05 * weights)
        else:
            next_weights = weights - cfg.optimizers.lr_weights * weight_grad
        next_lmb = jnp.maximum(0, lmb + cfg.optimizers.lr_lmb * jnp.sum(_main_grad * (cfg.training.beta*main_grad - weighted_aux_grad)))

        return next_params, next_weights, next_lmb, main_grad, weighted_aux_grad

    params = jnp.array(cfg.params)
    weights = jnp.array(cfg.weights)
    lmb = cfg.lmb

    training_history = {
        "params": [np.array(params)],
        "weights": [np.array(weights)],
        "lmbs": [np.array(lmb)],
        "main_grads": [],
        "weighted_aux_grads": [],
    }

    for i in trange(cfg.training.max_iter):
        params, weights, lmb, main_grad, weighted_aux_grad = update_step(params, weights, lmb)

        for key, value in zip(["params", "weights", "lmbs", "main_grads", "weighted_aux_grads"], [np.array(params), np.array(weights), np.array(lmb), np.array(-main_grad), np.array(-weighted_aux_grad)]):
            training_history[key].append(value)

    _, _, _, main_grad, weighted_aux_grad = update_step(params, weights, lmb)
    for key, value in zip(["main_grads", "weighted_aux_grads"], [np.array(-main_grad), np.array(-weighted_aux_grad)]):
            training_history[key].append(value)

    summary = {
        "params": [param.tolist() for param in training_history["params"]],
        "weights": [weight.tolist() for weight in training_history["weights"]],
        "lmbs": [lmb.tolist() for lmb in training_history["lmbs"]],
        "main_grads": [grad.tolist() for grad in training_history["main_grads"]],
        "weighted_aux_grads": [grad.tolist() for grad in training_history["weighted_aux_grads"]],
    }

    # Save summary
    summary_path = f"out/{cfg.experiment_name}/{cfg.sub_experiment_name}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    plot_lambda(cfg, training_history)
    plot2D(cfg, training_history)
    plot3D(cfg, training_history)


if __name__ == "__main__":
    main()