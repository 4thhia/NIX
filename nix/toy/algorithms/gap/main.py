import os
import json
import hydra
from omegaconf import DictConfig
from tqdm import trange

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from nix.toy.common.losses import create_loss_fn, create_loss_grad_fn, loss_fn_surface,
from nix.toy.common.utils import plot_lambda, plot2D, plot3D

@hydra.main(config_path="../_configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run training and evaluation."""
    os.makedirs(f"out/{cfg.experiment_name}/{cfg.sub_experiment_name}/{cfg.run_time}", exist_ok=True)

    main_loss_fns, main_grad_fns = create_loss_grad_fn(cfg.main.mux, cfg.main.muy, cfg.main.stdx, cfg.main.stdy, cfg.main.rho, cfg.main.flat)
    aux_loss_fns, aux_grad_fns = create_loss_grad_fn(cfg.aux.mux, cfg.aux.muy, cfg.aux.stdx, cfg.aux.stdy, cfg.aux.rho, cfg.aux.flat)

    def activation_fn(weights):
        if cfg.training.activation == "sigmoid":
            return nn.sigmoid(weights)
        elif cfg.training.activation == "tanh":
            return nn.tanh(weights)
        else:
            return weights

    def weight_regularization_fn(weights):
        regularization_loss = - cfg.training.regularization_coef * jnp.sum(weights * weights)
        return regularization_loss

    @jax.jit
    def update_step(params, weights, lmb, gamma_coef):
        main_loss = jnp.sum(jnp.array([fn(params) for fn in main_loss_fns]), axis=0)
        main_grad = jnp.sum(jnp.array([fn(params) for fn in main_grad_fns]), axis=0)
        norm = jnp.maximum(1e-8, jnp.linalg.norm(main_grad))
        main_grad_normalized = (1 - cfg.training.normalize) * main_grad + cfg.training.normalize * (main_grad / norm)

        aux_grads = [fn(params) for fn in aux_grad_fns]

        def loss_fn(weights):
            weighted_aux_grad = jnp.sum(jnp.array([activation_fn(w) * g for w, g in zip(weights, aux_grads)]), axis=0)
            weight_loss = - lmb * jnp.sum(main_grad_normalized * weighted_aux_grad) + weight_regularization_fn(weights)
            return weight_loss, weighted_aux_grad

        (_, weighted_aux_grad), weight_grad = jax.value_and_grad(loss_fn, has_aux=True)(weights)

        next_params = params - cfg.optimizers.lr_params * (main_grad + weighted_aux_grad)
        next_weights = weights - cfg.optimizers.lr_weights * (weight_grad)

        next_gamma_coef = gamma_coef + cfg.algorithm.gamma.coef.lr * (cfg.algorithm.target_loss - main_loss)
        next_gamma_coef = jnp.clip(next_gamma_coef, - cfg.algorithm.gamma.coef.bound, cfg.algorithm.gamma.coef.bound)
        gamma = (cfg.algorithm.target_loss > main_loss) * nn.sigmoid(next_gamma_coef) * cfg.algorithm.gamma.max

        lmb_gain   =  jnp.sum(main_grad_normalized * (cfg.algorithm.beta * main_grad - weighted_aux_grad))
        next_lmb = jnp.maximum(0, lmb + cfg.algorithm.lmb.lr * (lmb_gain - gamma))


        def total_loss_fn(params):
            main_loss = jnp.sum(jnp.array([fn(params) for fn in main_loss_fns]), axis=0)
            aux_loss = jnp.sum(weights * jnp.array([fn(params) for fn in aux_loss_fns]), axis=0)
            total_loss = main_loss + aux_loss
            return total_loss

        x_values = jnp.linspace(-1, 1, num=1000)
        pairs = jnp.stack([x_values, jnp.zeros_like(x_values)], axis=1)

        y_values = jax.vmap(total_loss_fn)(pairs)
        params_total_opt = x_values[jnp.argmin(y_values)]

        return next_params, activation_fn(next_weights), next_lmb, next_gamma_coef, main_loss, lmb_gain , gamma, main_grad, weighted_aux_grad, params_total_opt

    params = jnp.array(cfg.params)
    weights = jnp.array(cfg.weights)
    lmb = cfg.algorithm.lmb.initial_value
    gamma_coef = cfg.algorithm.gamma.coef.initial_value

    training_history = {
        "params": [np.array(params)],
        "weights": [np.array(weights)],
        "lmbs": [np.array(lmb)],
        "gamma_coef": [np.array(gamma_coef)],
        "main_loss": [],
        "lmb_gain": [] ,
        "gamma": [],
        "main_grads": [],
        "weighted_aux_grads": [],
        "params_total_opt": [],
    }

    for i in trange(cfg.training.max_iter):
        params, weights, lmb, gamma_coef, main_loss, lmb_gain , gamma, main_grad, weighted_aux_grad, params_total_opt = update_step(params, weights, lmb, gamma_coef)

        for key, value in zip(["params", "weights", "lmbs", "gamma_coef", "main_loss", "lmb_gain", "gamma", "main_grads", "weighted_aux_grads", "params_total_opt"], [np.array(params), np.array(weights), np.array(lmb), np.array(gamma_coef), np.array(main_loss), np.array(lmb_gain), np.array(gamma), np.array(-main_grad), np.array(-weighted_aux_grad), np.array(params_total_opt)]):
            training_history[key].append(value)

    _, _, _, _, main_loss, lmb_gain , gamma, main_grad, weighted_aux_grad, params_total_opt = update_step(params, weights, lmb, gamma_coef)
    for key, value in zip(["main_loss", "lmb_gain", "gamma", "main_grads", "weighted_aux_grads", "params_total_opt"], [np.array(main_loss), np.array(lmb_gain), np.array(gamma), np.array(-main_grad), np.array(-weighted_aux_grad), np.array(params_total_opt)]):
            training_history[key].append(value)

    summary = {
        "params": [param.tolist() for param in training_history["params"]],
        "weights": [weight.tolist() for weight in training_history["weights"]],
        "lmbs": [lmb.tolist() for lmb in training_history["lmbs"]],
        "gamma_coef": [gamma_coef.tolist() for gamma_coef in training_history["gamma_coef"]],
        "main_loss": [main_loss.tolist() for main_loss in training_history["main_loss"]],
        "lmb_gain": [lmb_gain.tolist() for lmb_gain in training_history["lmb_gain"]],
        "gamma": [gamma.tolist() for gamma in training_history["gamma"]],
        "main_grads": [grad.tolist() for grad in training_history["main_grads"]],
        "weighted_aux_grads": [grad.tolist() for grad in training_history["weighted_aux_grads"]],
        "params_total_opt": [params_total_opt.tolist() for params_total_opt in training_history["params_total_opt"]],
    }

    # Save summary
    summary_path = f"out/{cfg.experiment_name}/{cfg.sub_experiment_name}/{cfg.run_time}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f'Saved summary to out/{cfg.experiment_name}/{cfg.sub_experiment_name}/{cfg.run_time}/summary.json')


    plot_utils(cfg, training_history)
    plot2D(cfg, training_history)
    #plot3D(cfg, training_history)


if __name__ == "__main__":
    main()