import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Array, Float


class MLP(eqx.Module):
    layers: list
    for_bc: callable = eqx.field(static=True)

    def __init__(self, width: int, depth: int, for_bc: callable, key: PRNGKeyArray):
        layers = [2] + [width for _ in range(depth - 1)] + [5]
        keys = jr.split(key, len(layers) - 1)
        self.layers = [
            eqx.nn.Linear(_in, _out, key=_k)
            for _in, _out, _k in zip(layers[:-1], layers[1:], keys)
        ]
        self.for_bc = for_bc

    def __call__(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> Float[Array, "5"]:
        inputs = jnp.stack([xi1, xi2])
        for affine_transform in self.layers[:-1]:
            inputs = jnp.tanh(affine_transform(inputs))
        return self.layers[-1](inputs) * self.for_bc(xi1, xi2)

    def _u_and_theta(
        self,
        xi1: Float[Array, ""],
        xi2: Float[Array, ""],
        T: Float[Array, "3 3"] = jnp.eye(3),
    ) -> tuple[Float[Array, "3"], Float[Array, "2"]]:
        """
        T is the matrix (-a1-; -a2-; -a3-) for transforming u into its covariant component.
        """
        u, theta = jnp.split(self(xi1, xi2), [3])
        u = T @ u
        return u, theta

    def _u_and_theta_d(
        self,
        xi1: Float[Array, ""],
        xi2: Float[Array, ""],
        T: Float[Array, "3 3"] = jnp.eye(3),
    ) -> tuple[Float[Array, "3 2"], Float[Array, "2 2"]]:
        u_d1, theta_d1 = jax.jacfwd(self._u_and_theta, argnums=0)(xi1, xi2, T)
        u_d2, theta_d2 = jax.jacfwd(self._u_and_theta, argnums=1)(xi1, xi2, T)
        u_d = jnp.stack([u_d1, u_d2], 1)
        theta_d = jnp.stack([theta_d1, theta_d2], 1)
        return u_d, theta_d
