import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class LinearNagdhi:
    @staticmethod
    @jax.vmap
    def _membrane_strain(
        u: Float[Array, "3"],
        u_d: Float[Array, "3 2"],
        cov_II: Float[Array, "2 2"],
        christoffel_symbol: Float[Array, "2 2 2"],
    ) -> Float[Array, "2 2"]:
        """
        Membrane strain tensor.
        e_ab(u) = 0.5(u_{alpha | beta} + u_{beta | alpha}) - b_{alpha, beta} u_3
        Note that u_{alpha | beta} = u_{alpha, beta} - Christoffel(lambda, alpha, beta)u_lambda.
        """
        u_a_surf_b = u_d[:-1] - jnp.einsum(
            "lab, l -> ab", christoffel_symbol, u[:-1]
        )  # (2, 2)
        membrane_strain = 0.5 * (u_a_surf_b + u_a_surf_b.T) - cov_II * u[-1]
        return membrane_strain

    @staticmethod
    @jax.vmap
    def _bending_strain(
        u: Float[Array, "3"],
        u_d: Float[Array, "3 2"],
        theta: Float[Array, "2"],
        theta_d: Float[Array, "2 2"],
        mix_II: Float[Array, "2 2"],
        cov_III: Float[Array, "2 2"],
        christoffel_symbol: Float[Array, "2 2 2"],
    ) -> Float[Array, "2 2"]:
        """
        Bending strain tensor.
        """
        theta_a_surf_b = theta_d - jnp.einsum(
            "lab, l -> ab", christoffel_symbol, theta
        )  # (2, 2)
        first_term = theta_a_surf_b + theta_a_surf_b.T
        u_a_surf_b = u_d[:-1] - jnp.einsum(
            "lab, l -> ab", christoffel_symbol, u[:-1]
        )  # (2, 2)
        bu = u_a_surf_b.T @ mix_II
        second_term = bu + bu.T
        third_term = cov_III * u[-1]
        return 0.5 * (first_term - second_term) + third_term

    @staticmethod
    @jax.vmap
    def _shear_strain(
        u: Float[Array, "3"],
        u_d: Float[Array, "3 2"],
        theta: Float[Array, "2"],
        mix_II: Float[Array, "2 2"],
    ) -> Float[Array, "2"]:
        shear_strain = theta + u_d[2] + u[:2] @ mix_II
        return shear_strain
