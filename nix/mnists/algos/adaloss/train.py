def loss_balance(loss):
    loss = jnp.where(loss >= 0, jnp.log(loss + 1e-8), -jnp.log(-loss + 1e-8))
    return loss