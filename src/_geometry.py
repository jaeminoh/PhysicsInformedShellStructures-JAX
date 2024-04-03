import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float


class Geometry:
    """
    Index convention for mixed component: contravariant first, covariant last.

    Methods whose names start with _ are for single batch
    instance; ```jax.vmap```can vectorize them.

    All static methods can be re-used as utiliy functions.
    However, usually they will not be re-used after
    pre-computation of geometric quantities.
    """

    def __init__(
        self,
        xi1: Float[ArrayLike, "batch"],
        xi2: Float[ArrayLike, "batch"],
    ):
        """
        Pre-compute geometric quantities.
        Since these are only computed once, pure ```numpy```
        could be much faster than ```jax.numpy```. (ToDo)
        """
        # local base vectors
        self.cov_a1, self.cov_a2, self.cov_a3 = jax.vmap(self._cov_local_basis)(
            xi1, xi2
        )
        self.T = jnp.stack(
            [self.cov_a1, self.cov_a2, self.cov_a3], axis=1
        )  # transformation matrix: (batch, #cov_ai, 3)

        # partial derivative of tangent vectors
        self.tangents_d = jax.vmap(self._cov_tangents_d)(xi1, xi2)

        # fundamental forms on the mid surface
        self.cov_I = self._cov_I(self.cov_a1, self.cov_a2)
        self.sqrt_det_a = self._sqrt_det_a(self.cov_I)
        self.con_I = self._con_I(self.cov_I)
        self.cov_II = self._cov_II(self.tangents_d, self.cov_a3)
        self.mix_II = self._mix_II(self.con_I, self.cov_II)
        self.cov_III = self._cov_III(self.cov_II, self.mix_II)

        # christoffel symbol
        self.christoffel_symbol = self._christoffel_symbol(
            self.tangents_d, self.cov_a1, self.cov_a2, self.con_I
        )

        # area
        self.parametric_area = (xi2.max() - xi2.min()) * (xi1.max() - xi1.min())
        self.surface_area = self.parametric_area * self.sqrt_det_a.mean()

    def __call__(
        self, xi1: Float[ArrayLike, ""], xi2: Float[ArrayLike, ""]
    ) -> Float[ArrayLike, "3"]:
        """
        Chart for the mid-surface.
        (Hyperbolic Paraboloid here.)
        """
        x = xi1
        y = xi2
        z = xi1**2 - xi2**2
        return jnp.stack([x, y, z])

    def _cov_tangents(
        self, xi1: Float[ArrayLike, ""], xi2: Float[ArrayLike, ""]
    ) -> tuple[Float[ArrayLike, "3"]]:
        """
        Computing local covariant tangent vectors of the given chart.
        Will be ```jax.vmap```-ed.
        """
        cov_a1 = jax.jacfwd(self, argnums=0)(xi1, xi2)  # (3,)
        cov_a2 = jax.jacfwd(self, argnums=1)(xi1, xi2)  # (3,)
        return cov_a1, cov_a2

    def _cov_tangents_d(
        self, xi1: Float[ArrayLike, ""], xi2: Float[ArrayLike, ""]
    ) -> tuple[Float[ArrayLike, "2 2 3"]]:
        """
        output has a shape of (2, 2, 3).
        0th index: a1, a2
        1st index: partial_1, partial_2
        """
        cov_a1_d = jax.jacfwd(self._cov_tangents, argnums=0)(xi1, xi2)  # ((3,), (3,))
        cov_a1_d = jnp.stack(cov_a1_d)  # (2, 3)
        cov_a2_d = jax.jacfwd(self._cov_tangents, argnums=1)(xi1, xi2)  # ((3,), (3,))
        cov_a2_d = jnp.stack(cov_a2_d)  # (2, 3)
        tangents_d = jnp.stack([cov_a1_d, cov_a2_d])  # (2, 2, 3)
        return tangents_d

    def _cov_local_basis(
        self, xi1: Float[ArrayLike, ""], xi2: Float[ArrayLike, ""]
    ) -> tuple[Float[ArrayLike, "3"]]:
        """
        Computing local covariant base vectors (a1, a2, a3) of the given chart.
        """
        cov_a1, cov_a2 = self._cov_tangents(xi1, xi2)
        cov_a3 = self._from_tangents_to_normal(cov_a1, cov_a2)
        return cov_a1, cov_a2, cov_a3

    @staticmethod
    def _from_tangents_to_normal(a1: Float[ArrayLike, "3"], a2: Float[ArrayLike, "3"]):
        """
        Utility function, from tangent vectors to the unit normal vector.
        """
        a1_cross_a2 = jnp.cross(a1, a2)
        a3 = a1_cross_a2 / jnp.maximum(jnp.linalg.norm(a1_cross_a2), 1e-12)
        return a3

    @staticmethod
    @jax.vmap
    def _cov_I(
        cov_a1: Float[ArrayLike, "3"],
        cov_a2: Float[ArrayLike, "3"],
    ) -> Float[ArrayLike, "2 2"]:
        """
        Computing mid-surface metric tensor (a)_{alpha, beta} = a_alpha cdot a_beta
        * 3d local base vectors are denoted by g_i.
        """
        A = jnp.stack([cov_a1, cov_a2], 1)  # (3, 2)
        cov_I = A.T @ A
        return cov_I

    @staticmethod
    @jax.vmap
    def _sqrt_det_a(cov_I: Float[ArrayLike, "2 2"]) -> Float[ArrayLike, ""]:
        return jnp.sqrt(jnp.linalg.det(cov_I))

    @staticmethod
    @jax.vmap
    def _con_I(cov_I: Float[ArrayLike, "2 2"]) -> Float[ArrayLike, "2 2"]:
        """
        Getting contravariant components of the first fundamental form (metric tensor)
        from the covariant components.
        """
        con_I = jnp.linalg.solve(cov_I, jnp.eye(cov_I.shape[0]))
        return con_I

    @staticmethod
    @jax.vmap
    def _cov_II(
        tangents_d: Float[ArrayLike, "2 2 3"],
        cov_a3: Float[ArrayLike, "3"],
    ) -> Float[ArrayLike, "2 2"]:
        """
        Computing the covariant components of the second fundamental form
        (curvature tensor).
        """
        cov_II = jnp.einsum("...abz, z -> ab", tangents_d, cov_a3)
        return cov_II

    @staticmethod
    @jax.vmap
    def _mix_II(
        con_I: Float[ArrayLike, "2 2"], cov_II: Float[ArrayLike, "2 2"]
    ) -> Float[ArrayLike, "2 2"]:
        """
        Getting the mixed components of the second fundamental form
        (curvature tensor) from the covariant components
        by multiplying the first fundamental form.
        """
        mix_II = con_I @ cov_II
        return mix_II

    @staticmethod
    @jax.vmap
    def _cov_III(
        cov_II: Float[ArrayLike, "2 2"], mix_II: Float[ArrayLike, "2 2"]
    ) -> Float[ArrayLike, "2 2"]:
        """
        The third fundamental form (III)_{alpha, beta} = sum_{lambda} b^{lambda}_{alpha} b_{lambda, beta}.
        Note that the second fundamental form is a symmetric tensor.
        Hence the actual computation is done via cov_II @ mix_II.
        """
        III = cov_II @ mix_II
        return III

    @staticmethod
    @jax.vmap
    def _christoffel_symbol(
        tangents_d: Float[ArrayLike, "2 2 3"],
        cov_a1: Float[ArrayLike, "3"],
        cov_a2: Float[ArrayLike, "3"],
        con_I: Float[ArrayLike, "2 2"],
    ) -> Float[ArrayLike, "2 2 2"]:
        """
        Christoffel symbol Gamma(lambda, alpha, beta): a_{alpha, beta} dot a^lambda.
        """
        con_a = jnp.stack([cov_a1, cov_a2], 1) @ con_I
        christoffel_symbol = jnp.einsum("abz, zl -> lab", tangents_d, con_a)
        return christoffel_symbol


if __name__ == "__main__":
    # Build a test based on FEA-Shell, hyperbolic parabolid.
    pass