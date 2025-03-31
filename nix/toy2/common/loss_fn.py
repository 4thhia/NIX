import numpy as np
import jax
import jax.numpy as jnp


def create_loss_fn(mux, stdx, flat=0):
    def loss_fn(x):
        coef = 1 / (jnp.sqrt(2 * jnp.pi) * stdx)
        term1 = ((x - mux) / stdx) ** 2
        _z = - coef * jnp.exp(-(1 / 2) * term1)

        z_flat = jnp.maximum(_z, - 0.7 * coef)
        z = (1-flat) * _z + flat * z_flat
        return 10 * z
    return loss_fn


def create_loss_fns(mux, stdx, flat):
    loss_fns = []
    for i in range(len(mux)):
        loss_fn = create_loss_fn(mux[i], stdx[i], flat[i])
        loss_fns.append(loss_fn)

    return loss_fns