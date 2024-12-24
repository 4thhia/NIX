from typing import Callable, List, Union, Tuple
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


def create_loss_fn(mux, muy, stdx, stdy, rho, flat=False):
    def loss_fn(x):
        coef = 1 / (2 * jnp.pi * stdx * stdy * jnp.sqrt(1 - rho**2))
        term1 = ((x[0] - mux) / stdx) ** 2
        term2 = -2 * rho * (x[0] - mux) * (x[1] - muy) / (stdx * stdy)
        term3 = ((x[1] - muy) / stdy) ** 2
        z = - coef * jnp.exp(-1 / (2 * (1 - rho**2)) * (term1 + term2 + term3))

        if flat:
            z = jnp.maximum(z, - 0.7*coef)

        return 10 * z
    return loss_fn


def loss_fn_surface(mux, muy, stdx, stdy, rho, flat, lim, n=50):
    loss_fn, _ = create_loss_grad_fn(mux, muy, stdx, stdy, rho, flat)
    xs = np.linspace(-lim, lim, n)
    ys = xs.copy()

    surfaces = []
    for loss_fn_i in loss_fn:
        zs = np.zeros(shape=(xs.shape[0], ys.shape[0]))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                params = np.array([x, y])
                zs[i, j] = loss_fn_i(params)

        surfaces.append((xs, ys, zs))

    return surfaces


def create_loss_grad_fn(
    mux: Union[float, List[float]],
    muy: Union[float, List[float]],
    stdx: Union[float, List[float]],
    stdy: Union[float, List[float]],
    rho: Union[float, List[float]],
    flat: bool,
) -> Union[Tuple[Callable, Callable], List[Tuple[Callable, Callable]]]:

    loss_fns = []
    grad_fns = []
    for i in range(len(mux)):
        loss_fn_i = create_loss_fn(mux[i], muy[i], stdx[i], stdy[i], rho[i], flat)
        grad_fn_i = jax.grad(loss_fn_i)
        loss_fns.append(loss_fn_i)
        grad_fns.append(grad_fn_i)

    return loss_fns, grad_fns