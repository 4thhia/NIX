import numpy as np
import jax
import jax.numpy as jnp


def create_loss_fn(mux, muy, stdx, stdy, rho, flat=0):
    def loss_fn(x):
        coef = 1 / (2 * jnp.pi * stdx * stdy * jnp.sqrt(1 - rho**2))
        term1 = ((x[0] - mux) / stdx) ** 2
        term2 = -2 * rho * (x[0] - mux) * (x[1] - muy) / (stdx * stdy)
        term3 = ((x[1] - muy) / stdy) ** 2
        _z = - coef * jnp.exp(-1 / (2 * (1 - rho**2)) * (term1 + term2 + term3))

        z_flat = jnp.maximum(_z, - 0.7 * coef)
        z = (1-flat) * _z + flat * z_flat
        return 10 * z
    return loss_fn


def create_loss_grad_fn(mux, muy, stdx, stdy, rho, flat):
    loss_fns = []
    grad_fns = []
    for i in range(len(mux)):
        loss_fn = create_loss_fn(mux[i], muy[i], stdx[i], stdy[i], rho[i], flat[i])
        grad_fn = jax.grad(loss_fn)
        loss_fns.append(loss_fn)
        grad_fns.append(grad_fn)

    return loss_fns, grad_fns


def loss_fn_surface(mux, muy, stdx, stdy, rho, flat, lim, n=50):
    loss_fns, _ = create_loss_grad_fn(mux, muy, stdx, stdy, rho, flat)
    xs = np.linspace(-lim, lim, n)
    ys = xs.copy()

    surfaces = []
    for loss_fn in loss_fns:
        zs = np.zeros(shape=(xs.shape[0], ys.shape[0]))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                params = np.array([x, y])
                zs[i, j] = loss_fn(params)

        surfaces.append((xs, ys, zs))

    return surfaces