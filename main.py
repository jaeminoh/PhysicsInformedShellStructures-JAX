import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import jaxopt
import equinox as eqx
from scipy.stats.qmc import Sobol

from src._geometry import HyperbolicParaboloid
from src._linear_elastic import LinearElastic
from src._linear_nagdhi import LinearNagdhi
from src.nn import MLP


seed = 1
rng = jr.PRNGKey(seed)
xi_col = Sobol(d=2).random(2**11) - 0.5
xi1 = jnp.asarray(xi_col[:, 0])
xi2 = jnp.asarray(xi_col[:, 1])

print(f"precision: {xi1.dtype}")

print("Pre-computing geometric quantities...")
tic = time.time()
geom = HyperbolicParaboloid(xi1, xi2)

shell_model = LinearNagdhi()
material_model = LinearElastic(1.0, 0.3)

C = jax.vmap(material_model._C)(geom.con_I)  # precomputable
D = jax.vmap(material_model._D)(geom.con_I)  # precomputable
toc = time.time()
print(f"Done! Elapsed time: {toc - tic:.2f}s")
thickness = 0.1
shear_factor = 5.0 / 6.0


def fully_clamped(xi1, xi2):
    return (xi1**2 - 0.25) * (xi2**2 - 0.25)


rng, init_key = jr.split(rng)

pinn = MLP(50, 4, geom.T_u, for_bc=fully_clamped, key=init_key)


####################
# Energy calculation
####################
def surface_integral(
    _integrand, _jacobian=geom.sqrt_det_a, _factor=geom.parametric_area
):
    """
    Utility function for (Monte-Carlo) integration over a surface
    """
    value = (_integrand * _jacobian).mean() * _factor
    return value


def loss(pinn):
    uhat, u, u_d, theta, theta_d = jax.vmap(pinn._u_and_theta_d)(xi1, xi2)

    membrane_strain = shell_model._membrane_strain(
        u, u_d, geom.cov_II, geom.christoffel_symbol
    )
    bending_strain = shell_model._bending_strain(
        u, u_d, theta, theta_d, geom.mix_II, geom.cov_III, geom.christoffel_symbol
    )
    shear_strain = shell_model._shear_strain(u, u_d, theta, geom.mix_II)

    # external energy
    _external_energy = -1 * thickness * uhat[:, 2]  # gravity
    external_energy = surface_integral(_external_energy)  # correct

    # membrane energy
    _membrane_energy = jnp.einsum(
        "...ab,...abcd,...cd", membrane_strain, C, membrane_strain
    )
    membrane_energy = 0.5 * thickness * surface_integral(_membrane_energy)

    # bending energy
    _bending_energy = jnp.einsum(
        "...ab,...abcd,...cd", bending_strain, C, bending_strain
    )
    bending_energy = 0.5 * (thickness**3 / 12) * surface_integral(_bending_energy)

    # shear energy
    _shear_energy = jnp.einsum("...a,...ab,...b", shear_strain, D, shear_strain)
    shear_energy = 0.5 * thickness * shear_factor * surface_integral(_shear_energy)

    inner_energy = membrane_energy + bending_energy + shear_energy

    loss = inner_energy - external_energy
    return loss, (
        inner_energy,
        external_energy,
        membrane_energy / inner_energy,
        bending_energy / inner_energy,
        shear_energy / inner_energy,
    )


data = np.genfromtxt(
    "fem_sol/fenics_pred_hyperb_parab_fully_clamped.csv", delimiter=",", skip_header=1
)
xi1_ = data[:, 0]
xi2_ = data[:, 1]
ux = data[:, 2]
uy = data[:, 3]
uz = data[:, 4]
theta1 = data[:, 5]
theta2 = data[:, 6]


niter_adam = 10**3
opt = jaxopt.OptaxSolver(
    loss,
    optax.adam(optax.cosine_decay_schedule(1e-3, niter_adam)),
    maxiter=niter_adam,
    has_aux=True,
)
print("adam stage...")
tic = time.time()
adam_pinn, state = opt.run(pinn)
toc = time.time()
i, e, m, b, s = state.aux
print(
    f"""Done! Elapsed time: {toc - tic:.2f}s
    i: {i:.3e}, e: {e:.3e}, m: {m:.2f}, b: {b:.2f}, s: {s:.2f}"""
)
# save model parameters
eqx.tree_serialise_leaves("params/adam.eqx", adam_pinn)


opt = jaxopt.LBFGS(fun=loss, has_aux=True)

print("LBFGS running...")
state = opt.init_state(pinn)


@jax.jit
def step(pinn, state):
    pinn, state = opt.update(pinn, state)
    return pinn, state


tic = time.time()
min_loss = np.Inf
for it in range(1, 2000 + 1):
    pinn, state = step(pinn, state)
    if it % 100 == 0:
        energy = state.value
        if np.isnan(energy).sum() > 0:
            print("NaN!")
            break
        if energy < min_loss:
            min_loss = energy
            i, e, m, b, s = state.aux
            print(
                f"it: {it}, i: {i:.3e}, e: {e:.3e}, m: {m:.2f}, b: {b:.2f}, s: {s:.2f}"
            )
toc = time.time()
print(f"Done! Elapsed time: {toc - tic:.2f}s")

i, e, m, b, s = state.aux

print(
    f"inner: {i:.3e}, external: {e:.3e}, memb: {m:.2f}, bend: {b:.2f}, shear: {s:.2f}"
)

eqx.tree_serialise_leaves("params/lbfgs.eqx", pinn)
