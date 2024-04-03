import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class LinearElastic:
    def __init__(
        self, young_modulus: Float[Array, ""], poisson_ratio: Float[Array, ""]
    ):
        self.mu = 0.5 * (1 / (1 + poisson_ratio)) * young_modulus
        self.Lambda = poisson_ratio * young_modulus / (1 - poisson_ratio**2)

    def _C(self, con_I: Float[Array, "2 2"]) -> Float[Array, "2 2 2 2"]:
        C = (
            (2 * self.Lambda * self.mu)
            / (self.Lambda + 2 * self.mu)
            * jnp.einsum("ab, cd -> abcd", con_I, con_I)
            + self.mu * jnp.einsum("ac, bd -> abcd", con_I, con_I)
            + self.mu * jnp.einsum("ad, bc -> abcd", con_I, con_I)
        )
        return C

    def _D(self, con_I: Float[Array, "2 2"]):
        return con_I * self.mu
