import jax
import numpy as np
import jax.random as jr
import equinox as eqx

from src.nn import MLP
from src._geometry import HyperbolicParaboloid
from src._linear_elastic import LinearElastic
from src._linear_nagdhi import LinearNagdhi


def bmv(a, b):
    return np.einsum("...ij,...j", a, b)


def bdot(a, b):
    return np.einsum("...i,...i", a, b)


xx = np.linspace(-0.5, 0.5, 201)
mesh = np.meshgrid(xx, xx, indexing="ij")
xi1 = mesh[0].ravel()
xi2 = mesh[1].ravel()

geom = HyperbolicParaboloid(xi1, xi2)
material = LinearElastic(1.0, 0.3)
C_jax = jax.vmap(material._C)(geom.con_I)
D_jax = jax.vmap(material._D)(geom.con_I)

con_metric = geom.con_I
C = np.zeros((xi1.size, 3, 3))
C[:, 0, 0] = (
    material.Lambda * con_metric[:, 0, 0] ** 2
    + 2.0 * material.mu * con_metric[:, 0, 0] ** 2
)
C[:, 0, 1] = (
    material.Lambda * con_metric[:, 0, 0] * con_metric[:, 1, 1]
    + 2.0 * material.mu * con_metric[:, 0, 1] ** 2
)
C[:, 0, 2] = (
    material.Lambda * con_metric[:, 0, 0] * con_metric[:, 0, 1]
    + 2.0 * material.mu * con_metric[:, 0, 0] * con_metric[:, 0, 1]
)
C[:, 1, 0] = (
    material.Lambda * con_metric[:, 0, 0] * con_metric[:, 1, 1]
    + 2.0 * material.mu * con_metric[:, 0, 1] ** 2
)
C[:, 1, 1] = (
    material.Lambda * con_metric[:, 1, 1] ** 2
    + 2.0 * material.mu * con_metric[:, 1, 1] ** 2
)
C[:, 1, 2] = (
    material.Lambda * con_metric[:, 1, 1] * con_metric[:, 0, 1]
    + 2.0 * material.mu * con_metric[:, 1, 1] * con_metric[:, 0, 1]
)
C[:, 2, 0] = (
    material.Lambda * con_metric[:, 0, 0] * con_metric[:, 0, 1]
    + 2.0 * material.mu * con_metric[:, 0, 0] * con_metric[:, 0, 1]
)
C[:, 2, 1] = (
    material.Lambda * con_metric[:, 1, 1] * con_metric[:, 0, 1]
    + 2.0 * material.mu * con_metric[:, 1, 1] * con_metric[:, 0, 1]
)
C[:, 2, 2] = material.Lambda * con_metric[:, 0, 1] ** 2 + material.mu * (
    con_metric[:, 0, 0] * con_metric[:, 1, 1] + con_metric[:, 0, 1] ** 2
)

D = material.mu * con_metric


christoffel_sym = geom.christoffel_symbol
cov_curv = geom.cov_II

Bm = np.zeros((xi1.size, 3, 5))
Bm[:, 0, 0] = -christoffel_sym[:, 0, 0, 0]
Bm[:, 0, 1] = -christoffel_sym[:, 1, 0, 0]
Bm[:, 0, 2] = -cov_curv[:, 0, 0]
Bm[:, 1, 0] = -christoffel_sym[:, 0, 1, 1]
Bm[:, 1, 1] = -christoffel_sym[:, 1, 1, 1]
Bm[:, 1, 2] = -cov_curv[:, 1, 1]
Bm[:, 2, 0] = -christoffel_sym[:, 0, 0, 1] - christoffel_sym[:, 0, 1, 0]
Bm[:, 2, 1] = -christoffel_sym[:, 1, 1, 0] - christoffel_sym[:, 1, 0, 1]
Bm[:, 2, 2] = -2.0 * cov_curv[:, 0, 1]

Bm1 = np.zeros(
    (
        xi1.size,
        3,
        5,
    )
)
Bm1[:, 0, 0] = 1.0
Bm1[:, 2, 1] = 1.0

Bm2 = np.zeros((xi1.size, 3, 5))
Bm2[:, 1, 1] = 1.0
Bm2[:, 2, 0] = 1.0


christoffel_sym = geom.christoffel_symbol
mixed_curv = geom.mix_II
third_ff = geom.cov_III

Bk = np.zeros((xi1.size, 3, 5))
Bk[:, 0, 0] = (
    mixed_curv[:, 0, 0] * christoffel_sym[:, 0, 0, 0]
    + mixed_curv[:, 1, 0] * christoffel_sym[:, 0, 1, 0]
)
Bk[:, 0, 1] = (
    mixed_curv[:, 0, 0] * christoffel_sym[:, 1, 0, 0]
    + mixed_curv[:, 1, 0] * christoffel_sym[:, 1, 1, 0]
)
Bk[:, 0, 2] = third_ff[:, 0, 0]
Bk[:, 0, 3] = -christoffel_sym[:, 0, 0, 0]
Bk[:, 0, 4] = -christoffel_sym[:, 1, 0, 0]

Bk[:, 1, 0] = (
    mixed_curv[:, 0, 1] * christoffel_sym[:, 0, 0, 1]
    + mixed_curv[:, 1, 1] * christoffel_sym[:, 0, 1, 1]
)
Bk[:, 1, 1] = (
    mixed_curv[:, 0, 1] * christoffel_sym[:, 1, 0, 1]
    + mixed_curv[:, 1, 1] * christoffel_sym[:, 1, 1, 1]
)
Bk[:, 1, 2] = third_ff[:, 1, 1]
Bk[:, 1, 3] = -christoffel_sym[:, 0, 1, 1]
Bk[:, 1, 4] = -christoffel_sym[:, 1, 1, 1]

Bk[:, 2, 0] = (
    mixed_curv[:, 0, 1] * christoffel_sym[:, 0, 0, 0]
    + mixed_curv[:, 1, 1] * christoffel_sym[:, 0, 1, 0]
    + mixed_curv[:, 0, 0] * christoffel_sym[:, 0, 0, 1]
    + mixed_curv[:, 1, 0] * christoffel_sym[:, 0, 1, 1]
)
Bk[:, 2, 1] = (
    mixed_curv[:, 0, 1] * christoffel_sym[:, 1, 0, 0]
    + mixed_curv[:, 1, 1] * christoffel_sym[:, 1, 1, 0]
    + mixed_curv[:, 0, 0] * christoffel_sym[:, 1, 0, 1]
    + mixed_curv[:, 1, 0] * christoffel_sym[:, 1, 1, 1]
)
Bk[:, 2, 2] = 2.0 * third_ff[:, 0, 1]
Bk[:, 2, 3] = -christoffel_sym[:, 0, 0, 1] - christoffel_sym[:, 0, 1, 0]
Bk[:, 2, 4] = -christoffel_sym[:, 1, 0, 1] - christoffel_sym[:, 1, 1, 0]

Bk1 = np.zeros((xi1.size, 3, 5))
Bk1[:, 0, 0] = -mixed_curv[:, 0, 0]
Bk1[:, 0, 1] = -mixed_curv[:, 1, 0]
Bk1[:, 0, 3] = 1.0
Bk1[:, 2, 0] = -mixed_curv[:, 0, 1]
Bk1[:, 2, 1] = -mixed_curv[:, 1, 1]
Bk1[:, 2, 4] = 1.0

Bk2 = np.zeros((xi1.size, 3, 5))
Bk2[:, 1, 0] = -mixed_curv[:, 0, 1]
Bk2[:, 1, 1] = -mixed_curv[:, 1, 1]
Bk2[:, 1, 4] = 1.0
Bk2[:, 2, 0] = -mixed_curv[:, 0, 0]
Bk2[:, 2, 1] = -mixed_curv[:, 1, 0]
Bk2[:, 2, 3] = 1.0


By = np.zeros((xi1.size, 2, 5))
By[:, 0, 0] = mixed_curv[:, 0, 0]
By[:, 0, 1] = mixed_curv[:, 1, 0]
By[:, 0, 3] = 1.0
By[:, 1, 0] = mixed_curv[:, 0, 1]
By[:, 1, 1] = mixed_curv[:, 1, 1]
By[:, 1, 4] = 1.0

By1 = np.zeros((xi1.size, 2, 5))
By1[:, 0, 2] = 1.0

By2 = np.zeros((xi1.size, 2, 5))
By2[:, 1, 2] = 1.0


init_key = jr.PRNGKey(0)
pinn = MLP(50, 4, T_u=geom.T_u, key=init_key)

opt_pinn = eqx.tree_deserialise_leaves("params/adam.eqx", pinn)

uhat, u, u_d, theta, theta_d = jax.vmap(opt_pinn._u_and_theta_d)(xi1, xi2)
shell_model = LinearNagdhi()

pred_5d = np.concatenate([u, theta], 1)
pred_5d_1 = np.concatenate([u_d[..., 0], theta_d[..., 0]], 1)
pred_5d_2 = np.concatenate([u_d[..., 1], theta_d[..., 1]], 1)


# membrane energy
membrane_strains = bmv(Bm, pred_5d) + bmv(Bm1, pred_5d_1) + bmv(Bm2, pred_5d_2)
membrane_energy = bdot(membrane_strains, bmv(C, membrane_strains))

_membrane_strain = shell_model._membrane_strain(
    u, u_d, geom.cov_II, geom.christoffel_symbol
)
_membrane_energy = np.einsum(
    "...ab,...abcd,...cd", _membrane_strain, C_jax, _membrane_strain
)
print("membrane test", np.allclose(_membrane_energy, membrane_energy))
membrane_energy = (
    (membrane_energy * geom.sqrt_det_a).mean() * geom.parametric_area * 0.1 * 0.5
)
print(f"membrane energy: {membrane_energy:.3e}")
# bending energy
bending_strains = bmv(Bk, pred_5d) + bmv(Bk1, pred_5d_1) + bmv(Bk2, pred_5d_2)
bending_energy = bdot(bending_strains, bmv(C, bending_strains))

_bending_strains = shell_model._bending_strain(
    u, u_d, theta, theta_d, geom.mix_II, geom.cov_III, geom.christoffel_symbol
)
_bending_energy = np.einsum(
    "...ab,...abcd,...cd", _bending_strains, C_jax, _bending_strains
)
print("bending test", np.allclose(_bending_energy, bending_energy))
bending_energy = (
    (bending_energy * geom.sqrt_det_a).mean() * geom.parametric_area * 1e-3 / 12 * 0.5
)
print(f"bending energy: {bending_energy:.3e}")

# shear energy
shear_strains = bmv(By, pred_5d) + bmv(By1, pred_5d_1) + bmv(By2, pred_5d_2)
shear_energy = bdot(shear_strains, bmv(D, shear_strains))

_shear_strains = shell_model._shear_strain(u, u_d, theta, geom.mix_II)
_shear_energy = np.einsum("...a, ...ab, ...b", _shear_strains, D_jax, _shear_strains)
print("shear test", np.allclose(_shear_energy, shear_energy))
shear_energy = (
    (shear_energy * geom.sqrt_det_a).mean() * geom.parametric_area * 0.1 * 0.5 * 5 / 6
)
print(f"shear energy: {shear_energy:.3e}")

work = -1 * (uhat[:, 2] * geom.sqrt_det_a).mean() * geom.parametric_area * 0.1
print(f"work: {work:.3e}")
