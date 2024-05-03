import fire
import jax
import jax.random as jr
import equinox as eqx
import numpy as np

from src.nn import MLP
from src._geometry import HyperbolicParaboloid


def main(method: str = "lbfgs"):
    data = np.genfromtxt(
        "fem_sol/fenics_pred_hyperb_parab_fully_clamped.csv",
        delimiter=",",
        skip_header=1,
    )
    xi1 = data[:, 0]
    xi2 = data[:, 1]

    ux = data[:, 2]
    uy = data[:, 3]
    uz = data[:, 4]

    theta1 = data[:, 5]
    theta2 = data[:, 6]

    geom = HyperbolicParaboloid(xi1, xi2)
    init_key = jr.PRNGKey(0)
    pinn = MLP(50, 4, geom.T_u, key=init_key)
    pinn = eqx.tree_deserialise_leaves(f"params/{method}.eqx", pinn)

    (_, theta_p), u_p = jax.vmap(pinn)(xi1, xi2)

    def rel_l2(pred, true):
        return np.linalg.norm(pred - true) / np.linalg.norm(true)

    print(f"{method} results:")
    print(f"ux: {rel_l2(u_p[:,0], ux):.2e}")
    print(f"uy: {rel_l2(u_p[:,1], uy):.2e}")
    print(f"uz: {rel_l2(u_p[:,2], uz):.2e}")
    print(f"theta1: {rel_l2(theta_p[:,0], theta1):.2e}")
    print(f"theta2: {rel_l2(theta_p[:,1], theta2):.2e}")


if __name__ == "__main__":
    fire.Fire(main)
