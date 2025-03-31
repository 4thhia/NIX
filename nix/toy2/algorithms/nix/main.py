import os
import json
import hydra
from omegaconf import DictConfig
from tqdm import trange

import matplotlib.pyplot as plt
import imageio

import jax
import jax.numpy as jnp
import flax.linen as nn

from nix.toy2.common.loss_fn import create_loss_fns

@hydra.main(config_path="../_configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = f"out/{cfg.algorithm.name}/{cfg.experiment_name}/{cfg.sub_experiment_name}/{cfg.run_time}"
    os.makedirs(output_dir, exist_ok=True)

    print(f'cfg.training.max_iter:{cfg.training.max_iter}')


    def activation_fn(weight):
        if cfg.training.activation == "sigmoid":
            return nn.sigmoid(weight)
        elif cfg.training.activation == "tanh":
            return nn.tanh(weight)
        else:
            return weight

    def weight_regularization_fn(weight):
        regularization_loss = - cfg.training.regularization_coef * jnp.sum(weight * weight)
        return regularization_loss


    main_loss_fns = create_loss_fns(cfg.main.mux, cfg.main.stdx, cfg.main.flat)
    def main_loss_fn(x):
        main_loss = jnp.sum(jnp.array([loss_fn(x) for loss_fn in main_loss_fns]))
        return main_loss

    aux_loss_fns = create_loss_fns(cfg.aux.mux, cfg.aux.stdx, cfg.aux.flat)
    def aux_loss_fn(x):
        weighted_aux_loss = jnp.sum(jnp.array([loss_fn(x) for loss_fn in aux_loss_fns]))
        return weighted_aux_loss

    def weighted_aux_loss_fn(x, weight):
        weighted_aux_loss = jnp.sum(weight * jnp.array([loss_fn(x) for loss_fn in aux_loss_fns]))
        return weighted_aux_loss

    def total_loss_fn(x, weight):
        return main_loss_fn(x) + weighted_aux_loss_fn(x, weight)


    def train_step(x, weight, lmb):
        main_loss, main_grad = jax.value_and_grad(main_loss_fn)(x)
        norm = jnp.maximum(1e-8, jnp.linalg.norm(main_grad))
        main_grad_normalized = (1 - cfg.training.normalize) * main_grad + cfg.training.normalize * (main_grad / norm)

        aux_loss, aux_grad = jax.value_and_grad(weighted_aux_loss_fn)(x, weight)

        def weight_loss_fn(weight):
            aux_grad = jax.grad(weighted_aux_loss_fn, argnums=0)(x, weight)
            weight_loss = - lmb * jnp.mean(jnp.sum(main_grad_normalized * aux_grad, axis=-1)) + weight_regularization_fn(weight)
            return weight_loss

        weight_grad = jax.grad(weight_loss_fn)(weight)

        x -= cfg.optimizers.lr_params * (main_grad + aux_grad)
        weight -= cfg.optimizers.lr_weights * (weight_grad)
        lmb += jnp.maximum(0, cfg.optimizers.lr_lmb * jnp.sum(main_grad_normalized * (cfg.algorithm.beta*main_grad - aux_grad)))
        lmb = jnp.minimum(1, lmb)
        return x, activation_fn(weight), lmb, weight_loss_fn


    x = jnp.array(cfg.params)
    weight = jnp.array(cfg.weights)
    lmb = cfg.algorithm.lmb.initial_value

    frames = []


    for step in trange(cfg.training.max_iter):
        x, weight, lmb, weight_loss_fn = train_step(x, weight, lmb)

        # プロットの作成（縦2分割）
        fig, ax = plt.subplots(nrows=2, figsize=(6, 10))

        # x のプロット (上段)

        x_vals = jnp.linspace(-cfg.plot.x_lim, cfg.plot.x_lim, 300)
        total_vals = jax.vmap(total_loss_fn, in_axes=(0, None))(x_vals, weight)
        main_vals = jax.vmap(main_loss_fn)(x_vals)
        aux_vals = jax.vmap(aux_loss_fn)(x_vals)
        ax[0].plot(x_vals, total_vals, label="Total Loss Function")
        ax[0].plot(x_vals, main_vals, alpha=0.3, label="Main Loss Function")
        ax[0].plot(x_vals, aux_vals, alpha=0.3, label="Aux Loss Function")
        ax[0].scatter(x, total_loss_fn(x, weight), color='red', label="Current x")
        ax[0].set_title(fr"Algo: {cfg.algorithm.name} | Step: {step} - $\beta$: {cfg.algorithm.beta}")
        ax[0].set_xlim(-cfg.plot.x_lim, cfg.plot.x_lim)
        ax[0].legend()

        # weight のプロット (下段)
        weight_vals = jnp.linspace(-cfg.plot.weight_lim, cfg.plot.weight_lim, 300)
        y_vals2 = jax.vmap(weight_loss_fn)(weight_vals)
        ax[1].plot(weight_vals, y_vals2, label="Weight Loss Function")
        ax[1].scatter(weight, weight_loss_fn(weight), color='red', label="Current weight")
        ax[1].set_title(fr"Step: {step} - activation:{cfg.training.activation} | $\lambda$: {lmb}")
        ax[1].set_xlim(-cfg.plot.weight_lim, cfg.plot.weight_lim)
        ax[1].legend()

        # 画像保存
        frame_path = f"{output_dir}/step_{step:03d}.png"
        plt.savefig(frame_path)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    # GIF作成
    gif_path = f"{output_dir}/optimization_process.gif"
    imageio.mimsave(gif_path, frames, duration=0.2)

    print("Optimization finished. GIF saved as", gif_path)


if __name__ == "__main__":
    main()