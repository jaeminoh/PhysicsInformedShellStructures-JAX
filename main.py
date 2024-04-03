import jax
import jax.numpy as jnp
import jax.random as jr
import jaxopt
import numpy as np
import equinox as eqx

from src._geometry import Geometry
from src._linear_elastic import LinearElastic
from src._linear_nagdhi import LinearNagdhi
from src.nn import MLP

key = jr.PRNGKey(0)
key, *keys = jr.split(key, 3)
xi1 = jr.uniform(keys[0], (2048,)) - 0.5
xi2 = jr.uniform(keys[1], (2048,)) - 0.5

geom = Geometry(xi1, xi2)

shell_model = LinearNagdhi()
material_model = LinearElastic(1.0, 0.3)

C = jax.vmap(material_model._C)(geom.con_I)  # precomputable
D = jax.vmap(material_model._D)(geom.con_I)  # precomputable
thickness = 0.1
shear_factor = 5 / 6

key, init_key = jr.split(key)
pinn = MLP(50, 4, lambda xi1, xi2: (xi1**2 - 0.25) * (xi2**2 - 0.25), init_key)


####################
# Energy calculation
####################
def surface_integral(
    _integrand, _jacobian=geom.sqrt_det_a, _factor=geom.parametric_area
):
    value = (_integrand * _jacobian).mean() * _factor
    return value


def loss(pinn):
    u, theta = jax.vmap(pinn._u_and_theta)(xi1, xi2, geom.T)  # (batch, 3), (batch, 2)
    u_d, theta_d = jax.vmap(pinn._u_and_theta_d)(
        xi1, xi2, geom.T
    )  # (batch, #u_i, 2), (batch, #theta_i, 2)

    membrane_strain = shell_model._membrain_strain(
        u, u_d, geom.cov_II, geom.christoffel_symbol
    )
    bending_strain = shell_model._bending_strain(
        u, u_d, theta, theta_d, geom.mix_II, geom.cov_III, geom.christoffel_symbol
    )
    shear_strain = shell_model._shear_strain(u, u_d, theta, geom.mix_II)

    # energy integrand
    _membrane_energy = (
        jnp.einsum("...ab, ...abcd, ...cd", membrane_strain, C, membrane_strain)
        * 0.5
        * thickness
    )
    _bending_energy = (
        jnp.einsum("...ab, ...abcd, ...cd", bending_strain, C, bending_strain)
        * 0.5
        * (thickness**3 / 12)
    )
    _shear_energy = (
        jnp.einsum("...a, ...ab, ...b", shear_strain, D, shear_strain)
        * 0.5
        * thickness
        * shear_factor
    )
    uhat, _ = jax.vmap(pinn._u_and_theta)(xi1, xi2)
    _external_energy = -1 * thickness * uhat[:, -1]
    external_energy = surface_integral(_external_energy)

    membrane_energy = surface_integral(_membrane_energy)

    bending_energy = surface_integral(_bending_energy)

    shear_energy = surface_integral(_shear_energy)

    inner_energy = membrane_energy + bending_energy + shear_energy

    loss = inner_energy - external_energy
    return loss, (
        membrane_energy / inner_energy,
        bending_energy / inner_energy,
        shear_energy / inner_energy,
    )


opt = jaxopt.LBFGS(loss, has_aux=True, maxiter=100, linesearch="backtracking")
opt_pinn, state = opt.run(pinn)

final_loss = state.value
m, b, s = state.aux

print(f"toal: {final_loss:.3e}, memb: {m:.3e}, bend: {b:.3e}, shear: {s:.3e}")


eqx.tree_serialise_leaves("params/weak.eqx", opt_pinn)

print(opt_pinn.layers[0].bias)