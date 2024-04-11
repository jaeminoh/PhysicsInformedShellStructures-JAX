import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float


class Geometry:
    """
    Base class for Geometry.

    Index convention for mixed component: contravariant first, covariant last.

    Methods whose names start with _ are for single batch
    instance; ```jax.vmap```can vectorize them.

    All static methods can be re-used as utiliy functions.
    However, usually they will not be re-used after
    pre-computation of geometric quantities.
    """

    def __init__(
        self,
        xi1: Float[ArrayLike, " batch"],
        xi2: Float[ArrayLike, " batch"],
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
        # partial derivative of tangent vectors
        self.tangents_d = jax.vmap(self._cov_tangents_d)(xi1, xi2)

        # fundamental forms of the mid surface
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
        Subclass must define this.
        """
        raise NotImplementedError

    def _cov_tangents(
        self, xi1: Float[ArrayLike, ""], xi2: Float[ArrayLike, ""]
    ) -> tuple[Float[ArrayLike, "3"]]:
        """
        Computing local covariant tangent vectors of the given chart.
        Will be ```jax.vmap```'ed.
        """
        cov_a1 = jax.jacfwd(self, argnums=0)(xi1, xi2)  # (3,)
        cov_a2 = jax.jacfwd(self, argnums=1)(xi1, xi2)  # (3,)
        return cov_a1, cov_a2

    def _cov_tangents_d(
        self, xi1: Float[ArrayLike, ""], xi2: Float[ArrayLike, ""]
    ) -> tuple[Float[ArrayLike, "2 2 3"]]:
        """
        0th index: a1, a2
        1st index: partial_1, partial_2
        """
        cov_a_d1 = jax.jacfwd(self._cov_tangents, argnums=0)(xi1, xi2)  # ((3,), (3,))
        cov_a_d1 = jnp.stack(cov_a_d1)  # (2, 3)
        cov_a_d2 = jax.jacfwd(self._cov_tangents, argnums=1)(xi1, xi2)  # ((3,), (3,))
        cov_a_d2 = jnp.stack(cov_a_d2)  # (2, 3)
        tangents_d = jnp.stack([cov_a_d1, cov_a_d2], axis=1)  # (2, 2, 3)
        return tangents_d

    @staticmethod
    def _from_tangents_to_normal(a1: Float[ArrayLike, "3"], a2: Float[ArrayLike, "3"]):
        """
        Utility function, from tangent vectors to the unit normal vector.
        """
        a1_cross_a2 = jnp.cross(a1, a2)
        a3 = a1_cross_a2 / jnp.maximum(jnp.linalg.norm(a1_cross_a2), 1e-12)
        return a3

    def _cov_local_basis(
        self, xi1: Float[ArrayLike, ""], xi2: Float[ArrayLike, ""]
    ) -> tuple:
        """
        Computing local covariant base vectors (a1, a2, a3) of the given chart.
        """
        cov_a1, cov_a2 = self._cov_tangents(xi1, xi2)
        cov_a3 = self._from_tangents_to_normal(cov_a1, cov_a2)
        return cov_a1, cov_a2, cov_a3

    def T_u(self, xi1, xi2):
        return jnp.stack(self._cov_local_basis(xi1, xi2))

    @staticmethod
    @jax.vmap
    def _cov_I(
        cov_a1: Float[ArrayLike, "3"],
        cov_a2: Float[ArrayLike, "3"],
    ) -> Float[ArrayLike, "2 2"]:
        """
        Computing mid-surface metric tensor (a)_{alpha, beta} = a_alpha cdot a_beta
            * 3d local base vectors are usually denoted by g_i.
            * symmetric.
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
        from the covariant components. Inverse matrix.
        """
        con_I = jnp.linalg.inv(cov_I)
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
            * symmetric.
        """
        cov_II = jnp.einsum("abz, z -> ab", tangents_d, cov_a3)
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
            * symmetric.
        """
        cov_III = (cov_II.T @ mix_II).T
        return cov_III

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
        con_a = con_I @ jnp.stack([cov_a1, cov_a2])  # (2, 3)
        christoffel_symbol = jnp.einsum("abz, lz -> lab", tangents_d, con_a)
        return christoffel_symbol


class HyperbolicParaboloid(Geometry):
    def __init__(self, xi1, xi2):
        super().__init__(xi1, xi2)

    def __call__(
        self, xi1: Float[ArrayLike, ""], xi2: Float[ArrayLike, ""]
    ) -> Float[ArrayLike, "3"]:
        """
        Chart for the mid-surface of Hyperbolic Paraboloid.
        """
        x = xi1
        y = xi2
        z = x**2 - y**2
        return jnp.stack([x, y, z])


class Plate(Geometry):
    def __init__(self, xi1, xi2):
        super().__init__(xi1, xi2)

    def __call__(
        self, xi1: Float[ArrayLike, ""], xi2: Float[ArrayLike, ""]
    ) -> Float[ArrayLike, "3"]:
        """
        Chart for the mid-surface of a plate.
        """
        x = xi1
        y = xi2
        z = 0.0
        return jnp.stack([x, y, z])


class Hemisphere(Geometry):
    def __init__(self, xi1, xi2):
        super().__init__(xi1, xi2)

    def __call__(self, xi1, xi2):
        """
        Chart for the mid-surface of a raidus 1 upper hemisphere.
        """
        x = xi1
        y = xi2
        z = jnp.linalg.norm(1.0 - x**2 - y**2)
        return jnp.stack([x, y, z])


class Cylinder(Geometry):
    def __init__(self, xi1, xi2):
        super().__init__(xi1, xi2)

    def __call__(self, xi1, xi2):
        """
        Chart for the mid-surface of a cylinder.
        """
        r = 1.0
        x = xi1
        _y = (xi2 - jnp.pi / 2) * r
        y = r * jnp.sin(_y / r)
        z = r * jnp.cos(_y / r)
        return jnp.stack([x, y, z])


if __name__ == "__main__":
    import numpy as np

    class Test(Geometry):
        def __init__(self, xi1, xi2):
            super().__init__(xi1, xi2)

        def __call__(
            self, xi1: Float[ArrayLike, ""], xi2: Float[ArrayLike, ""]
        ) -> Float[ArrayLike, "3"]:
            """
            Chart for the mid-surface.
            """
            x = xi1
            y = xi2
            z = xi1**2 - xi2**2
            return jnp.stack([x, y, 0.5 * z])

        @staticmethod
        def a(xi1, xi2):
            return 1 + xi1**2 + xi2**2

        def tangents_d_cov(self, xi1, xi2):
            a1_1 = jnp.array([0, 0, 1])
            a1_2 = jnp.zeros((3,))
            a2_1 = a1_2.copy()
            a2_2 = jnp.array([0, 0, -1])
            a1_d = jnp.stack([a1_1, a1_2])
            a2_d = jnp.stack([a2_1, a2_2])
            return jnp.stack([a1_d, a2_d])

        def christofell(self, xi1, xi2):
            a = self.a(xi1, xi2)
            gamma_111 = xi1 / a
            gamma_112 = 0.0
            gamma_121 = 0.0
            gamma_122 = -xi1 / a
            gamma_211 = -xi2 / a
            gamma_212 = 0.0
            gamma_221 = 0.0
            gamma_222 = xi2 / a
            christoffel = jnp.stack(
                [
                    jnp.array([[gamma_111, gamma_112], [gamma_121, gamma_122]]),
                    jnp.array([[gamma_211, gamma_212], [gamma_221, gamma_222]]),
                ]
            )
            return christoffel

        def II_cov(self, xi1, xi2):
            a = self.a(xi1, xi2)
            return jnp.diag(jnp.array([1 / a**0.5, -1 / a**0.5]))

        def II_mix(self, xi1, xi2):
            a = self.a(xi1, xi2)
            b11 = (1 + xi2**2) / a**1.5
            b12 = -xi1 * xi2 / a**1.5  # not symmetric.
            b21 = xi1 * xi2 / a**1.5
            b22 = -(1 + xi1**2) / a**1.5
            return jnp.array([[b11, b12], [b21, b22]])

        def I_con(self, xi1, xi2):
            a = self.a(xi1, xi2)
            I11 = 1 + xi2**2
            I12 = xi1 * xi2
            I21 = xi1 * xi2
            I22 = 1 + xi1**2
            return jnp.array([[I11, I12], [I21, I22]]) / a

        def a3(self, xi1, xi2):
            a = self.a(xi1, xi2)
            return jnp.array([-xi1, xi2, 1]) / a**0.5

        def III(self, xi1, xi2):
            a = self.a(xi1, xi2)
            III = jnp.array([[1 + xi2**2, -xi1 * xi2], [-xi1 * xi2, 1 + xi1**2]]) / a**2
            return III

    from scipy.stats.qmc import Sobol

    xi_col = Sobol(d=2).random_base2(14) - 0.5
    # xi = np.linspace(-0.5, 0.5, 200)
    # mesh = np.meshgrid(xi, xi, indexing="ij")
    # xi1 = mesh[0].ravel()
    # xi2 = mesh[1].ravel()
    xi1 = xi_col[:, 0]
    xi2 = xi_col[:, 1]
    geom = Test(xi1, xi2)

    # Tests
    print(
        "Test - normal vector:", np.allclose(geom.cov_a3, jax.vmap(geom.a3)(xi1, xi2))
    )
    print(
        "Test - jacobian:",
        np.allclose(geom.sqrt_det_a, jax.vmap(geom.a)(xi1, xi2) ** 0.5),
    )
    print(
        "Test - First fundamental form, contravariant:",
        np.allclose(geom.con_I, jax.vmap(geom.I_con)(xi1, xi2)),
    )
    print(
        "Test - derivative for the basis vectors:",
        np.allclose(geom.tangents_d, jax.vmap(geom.tangents_d_cov)(xi1, xi2)),
    )
    print(
        "Test - second fundamental form, covariant:",
        np.allclose(geom.cov_II, jax.vmap(geom.II_cov)(xi1, xi2)),
    )
    print(
        "Test - second fundamental form, mixed:",
        np.allclose(geom.mix_II, jax.vmap(geom.II_mix)(xi1, xi2)),
    )
    print(
        "Test - Christofell symbol:",
        np.allclose(geom.christoffel_symbol, jax.vmap(geom.christofell)(xi1, xi2)),
    )
    print(
        "Test - Third fundamental form, covariant:",
        np.allclose(geom.cov_III, jax.vmap(geom.III)(xi1, xi2)),
    )

    class SphericalCurve(Geometry):
        def __init__(self, u, v):
            super().__init__(u, v)

        def __call__(self, u, v):
            return jnp.array(
                [jnp.cos(u) * jnp.sin(v), jnp.sin(u) * jnp.sin(v), jnp.cos(v)]
            )

        def I_cov(self, u, v):
            return jnp.array([[jnp.sin(v) ** 2, 0.0], [0.0, 1.0]])

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    mesh = np.meshgrid(u, v, indexing="ij")
    u = mesh[0].ravel()
    v = mesh[1].ravel()
    geom = SphericalCurve(u, v)
    print(
        "Test - first fundamental form, covariant:",
        np.allclose(geom.cov_I, jax.vmap(geom.I_cov)(u, v)),
    )
