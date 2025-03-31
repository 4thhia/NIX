import os
import json
import hydra
from omegaconf import DictConfig
from tqdm import trange

import matplotlib.pyplot as plt
import imageio

import jax.numpy as jnp
import jax

from nix.toy2.common.loss_fn import create_loss_fns

@hydra.main(config_path="../_configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = f"out/{cfg.algorithm.name}/{cfg.experiment_name}/{cfg.sub_experiment_name}/{cfg.run_time}"
    os.makedirs(output_dir, exist_ok=True)

    main_loss_fns = create_loss_fns(cfg.main.mux, cfg.main.stdx, cfg.main.flat)
    def main_loss_fn(x):
        main_loss = jnp.sum(jnp.array([loss_fn(x) for loss_fn in main_loss_fns]))
        return main_loss

    def train_step(x):
        main_loss, main_grad = jax.value_and_grad(main_loss_fn)(x)
        x -= cfg.optimizers.lr_params * main_grad
        return x, main_loss

    x = jnp.array(cfg.params)

    frames = []

    for step in trange(cfg.training.max_iter):
        x, main_loss = train_step(x)

        # プロット
        fig, ax = plt.subplots()
        x_vals = jnp.linspace(-cfg.plot.lim, cfg.plot.lim, 300)
        y_vals = jax.vmap(main_loss_fn)(x_vals)
        ax.plot(x_vals, y_vals, label="Loss function")
        ax.scatter(x, main_loss_fn(x), color='red', label="Current x")
        ax.set_title(f"Step {step}")
        ax.set_xlim(-cfg.plot.lim, cfg.plot.lim)
        ax.legend()

        # 画像保存
        frame_path = f"{output_dir}/step_{step:03d}.png"
        plt.savefig(frame_path)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    # GIF作成
    gif_path = f"{output_dir}/optimization_process.gif"
    imageio.mimsave(gif_path, frames, duration=0.2)


    print("Optimization finished. GIF saved as 'optimization_process.gif'")


if __name__ == "__main__":
    main()