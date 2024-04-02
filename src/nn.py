import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray


class MLP(eqx.Module):
    layers: list

    def __init__(self, width: int, depth: int, key: PRNGKeyArray):
        layers = [3] + [width for _ in range(depth - 1)] + [5]
        keys = jr.split(key, len(layers) - 1)
        self.layers = [
            eqx.nn.Linear(_in, _out, _k)
            for _in, _out, _k in zip(layers[:-1], layers[1:], keys)
        ]

    def __call__(self, inputs):
        for affine_transform in self.layers[:-1]:
            inputs = jnp.tanh(affine_transform(inputs))
        return self.layers[-1](inputs)

