import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


class Geometry:
    """
    _methods are for single batch instance. To be vmap-ed.
    """

    def chart(self, xi1: Float[Array, ""], xi2: Float[Array, ""]) -> Float[Array, "3"]:
        """
        Hyperbolic Paraboloid.
        """
        x = xi1
        y = xi2
        z = xi1**2 - xi2**2
        return jnp.stack([x, y, z])

    @staticmethod
    def _from_tangents_to_normal(a1: Float[Array, "3"], a2: Float[Array, "3"]):
        a1_cross_a2 = jnp.cross(a1, a2)
        a3 = a1_cross_a2 / jnp.maximum(jnp.linalg.norm(a1_cross_a2), 1e-12)
        return a3

    def _cov_tangents(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> tuple[Float[Array, "3"]]:
        """
        Computing local covariant tangent vectors of the given chart.
        Will be ```jax.vmap```-ed.
        """
        cov_a1 = jax.jacfwd(self.chart, argnums=0)(xi1, xi2)
        cov_a2 = jax.jacfwd(self.chart, argnums=1)(xi1, xi2)
        return cov_a1, cov_a2

    def _cov_local_basis(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> tuple[Float[Array, "3"]]:
        """
        Computing local covariant base vectors (a1, a2, a3) of the given chart.
        """
        cov_a1, cov_a2 = self._cov_tangents(xi1, xi2)
        # cov_a3_unnorm = jnp.cross(cov_a1, cov_a2)
        # cov_a3 = cov_a3_unnorm / jnp.maximum(jnp.linalg.norm(cov_a3_unnorm), 1e-12)
        cov_a3 = self._from_tangents_to_normal(cov_a1, cov_a2)
        return cov_a1, cov_a2, cov_a3

    def _cov_metric_tensor(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> Float[Array, "2 2"]:
        """
        Computing (local) metric tensor:
            .. math::(g)_{ij} = a_i \cdot a_j
        """
        cov_a1, cov_a2 = self._cov_tangents(xi1, xi2)
        A = jnp.stack([cov_a1, cov_a2], 1)  # (3, 2)
        cov_I = A.T @ A
        return cov_I

    def _sqrt_det_a(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> Float[Array, ""]:
        cov_I = self._cov_metric_tensor(xi1, xi2)
        return jnp.sqrt(jnp.linalg.det(cov_I))

    def _con_metric_tensor(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> Float[Array, "2 2"]:
        cov_I = self._cov_metric_tensor(xi1, xi2)
        con_I = jnp.linalg.solve(cov_I, jnp.eye(cov_I.shape[0]))
        return con_I

    def _cov_local_basis_d(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> tuple[Float[Array, "3"]]:
        cov_a1_1, cov_a1_2 = jax.jacfwd(self._cov_tangents, argnums=0)(xi1, xi2)
        cov_a2_1, cov_a2_2 = jax.jacfwd(self._cov_tangents, argnums=1)(xi1, xi2)
        return cov_a1_1, cov_a1_2, cov_a2_1, cov_a2_2

    def _cov_curv_tensor(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> Float[Array, "2 2"]:
        cov_a_d = jnp.stack(self._cov_local_basis_d(xi1, xi2)).reshape(2, 2, 3)
        *_, cov_a3 = self._cov_local_basis(xi1, xi2)
        cov_II = jnp.einsum("...abz, z -> ab", cov_a_d, cov_a3)
        return cov_II

    def _mix_curv_tensor(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> Float[Array, "2 2"]:
        con_I = self._con_metric_tensor(xi1, xi2)
        cov_II = self._cov_curv_tensor(xi1, xi2)
        mix_II = con_I @ cov_II
        return mix_II

    def _cov_third(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> Float[Array, "2 2"]:
        """
        The third fundamental form (III)_{alpha, beta} = sum_{lambda} b^{lambda}_{alpha} b_{lambda, beta}.
        Note that the second fundamental form is a symmetric tensor.
        Hence the actual computation is done via cov_II @ mix_II.
        """
        cov_II = self._cov_curv_tensor(xi1, xi2)
        mix_II = self._mix_curv_tensor(xi1, xi2)
        III = cov_II @ mix_II
        return III

    def _christoffel_symbol(
        self, xi1: Float[Array, ""], xi2: Float[Array, ""]
    ) -> Float[Array, "2 2 2"]:
        cov_a1, cov_a2, _ = self._cov_local_basis(xi1, xi2)
        con_I = self._con_metric_tensor(xi1, xi2)
        con_a = jnp.stack([cov_a1, cov_a2], 1) @ con_I
        cov_a_d = jnp.stack(self._cov_local_basis_d(xi1, xi2)).reshape(2, 2, 3)
        christoffel_symbol = jnp.einsum("abz, zl -> lab", cov_a_d, con_a)
        return christoffel_symbol


if __name__ == "__main__":
    import jax.random as jr

    key1, key2 = jr.split(jr.PRNGKey(0))
    xi1 = jr.uniform(key1, (128,)) - 0.5
    xi2 = jr.uniform(key2, (128,)) - 0.5
    geom = Geometry()
    III = jax.vmap(geom._cov_third)(xi1, xi2)
    print(np.isclose(III[:,0,1], III[:,1,0]).mean())
