import jax.random as jr
from src.nn import MLP
import equinox as eqx

init_key = jr.PRNGKey(0)
pinn = MLP(50, 4, lambda xi1, xi2: (xi1**2 - 0.25) * (xi2**2 - 0.25), init_key)

opt_pinn = eqx.tree_deserialise_leaves("params/weak.eqx", pinn)

print(opt_pinn.layers[0].bias)