import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Array, Float


class MLP(eqx.Module):
    layers: list
    for_bc: callable = eqx.field(static=True)
    T_u: callable = eqx.field(static=True)

    def __init__(
        self,
        width: int,
        depth: int,
        T_u: callable,
        *,
        for_bc: callable = lambda x, y: (x**2 - 0.25) * (y**2 - 0.25),
        key: PRNGKeyArray,
    ):
        layers = [2] + [width for _ in range(depth - 1)] + [5]
        keys = jr.split(key, depth)
        self.layers = [
            eqx.nn.Linear(_in, _out, key=_k)
            for _in, _out, _k in zip(layers[:-1], layers[1:], keys)
        ]
        self.T_u = T_u
        self.for_bc = for_bc

    def __call__(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> Float[Array, "5"]:
        # Network forward pass + Hard constraint for Dirichlet B.C.
        inputs = jnp.stack([xi1, xi2])
        for layer in self.layers[:-1]:
            inputs = jnp.tanh(layer(inputs))
        nn = self.layers[-1](inputs) * self.for_bc(xi1, xi2)
        uhat, theta = jnp.split(nn, [3])
        u = self.T_u(xi1, xi2) @ uhat
        return (u, theta), uhat

    def _u_and_theta_d(
        self,
        xi1: Float[Array, ""],
        xi2: Float[Array, ""],
    ) -> tuple[Float[Array, "3 2"], Float[Array, "2 2"]]:
        (u, theta), (u_d1, theta_d1), uhat = jax.jvp(
            lambda xi1: self(xi1, xi2), (xi1,), (jnp.ones(xi1.shape),), has_aux=True
        )
        (u_d2, theta_d2) = jax.jvp(
            lambda xi2: self(xi1, xi2), (xi2,), (jnp.ones(xi2.shape),), has_aux=True
        )[1]
        u_d = jnp.stack([u_d1, u_d2], 1)
        theta_d = jnp.stack([theta_d1, theta_d2], 1)
        return uhat, u, u_d, theta, theta_d


if __name__ == "__main__":
    pass
