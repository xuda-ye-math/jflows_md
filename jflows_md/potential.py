"""Bundle-driven molecular potentials."""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from jflows.potential import Potential

from .core.coordinates import Internal_Coordinates
from .core.forcefield import Amber_OBC_Force_Field
from .source import Molecular_Source
from .system import Molecular_Bundle


KB_KJ_MOL_K = 0.00831446261815324

__all__ = ["KB_KJ_MOL_K", "Molecular_Potential", "Regularized_Molecular_Potential"]


class Molecular_Potential(Potential):
    """Physical reduced potential ``beta E(x(q)) - log J(q)``.

    ``regularized((e, r))`` returns the two-parameter regularized potential
    ``U^{rho}`` of the manuscript; this class itself carries no energy
    regularization, and a non-finite force-field value stays non-finite
    (the importance-weight screen gives it weight zero).
    """

    forcefield: Amber_OBC_Force_Field
    coordinates: Internal_Coordinates
    reference_positions_nm: Array
    source_mean: Array
    source_variance: Array
    beta: Array
    temperature_kelvin: float = eqx.field(static=True)
    bundle_temperature_kelvin: float = eqx.field(static=True)
    bundle_name: str = eqx.field(static=True)
    bundle_path: str = eqx.field(static=True)

    def __init__(
        self,
        bundle: Molecular_Bundle,
        *,
        temperature_kelvin: float | None = None,
    ):
        self.forcefield = Amber_OBC_Force_Field(bundle.system)
        self.coordinates = Internal_Coordinates(bundle.coordinates)
        self.reference_positions_nm = jnp.asarray(bundle.validation["frames_nm"][0])
        self.source_mean = jnp.asarray(bundle.coordinates["source_mean"])
        self.source_variance = jnp.asarray(bundle.coordinates["source_variance"])
        self.bundle_temperature_kelvin = float(bundle.manifest["temperature_kelvin"])
        self.temperature_kelvin = (
            self.bundle_temperature_kelvin
            if temperature_kelvin is None
            else float(temperature_kelvin)
        )
        self.beta = jnp.asarray(1.0 / (KB_KJ_MOL_K * self.temperature_kelvin))
        self.bundle_name = bundle.name
        self.bundle_path = str(bundle.path)

    @classmethod
    def from_bundle(
        cls,
        path_or_name: str | Path,
        *,
        root: str | Path | None = None,
        verify: bool = True,
        temperature_kelvin: float | None = None,
    ) -> "Molecular_Potential":
        return cls(
            Molecular_Bundle.load(path_or_name, root=root, verify=verify),
            temperature_kelvin=temperature_kelvin,
        )

    @property
    def domain(self):
        return self.coordinates.domain

    @property
    def dimension(self) -> int:
        return self.domain.dimension

    def cartesian(self, q: Array) -> Array:
        return self.coordinates.to_cartesian(q)[0]

    def physical_energy(self, q: Array) -> Array:
        return self.forcefield(self.cartesian(q))

    def energy_terms(self, q: Array) -> dict[str, Array]:
        x, logdet = self.coordinates.to_cartesian(q)
        terms = self.forcefield.energy_terms(x)
        reduced = {name: self.beta * value for name, value in terms.items()}
        reduced["logdet"] = logdet
        reduced["potential"] = reduced["total"] - logdet
        return reduced

    def reduced_energy(self, q: Array) -> Array:
        """Reduced energy ``beta E(x(q))`` without the Jacobian."""
        return self.beta * self.physical_energy(q)

    def __call__(self, q: Array) -> Array:
        x, logdet = self.coordinates.to_cartesian(q)
        return self.beta * self.forcefield(x) - logdet

    def reference_internal(self) -> Array:
        return self.coordinates.to_internal(self.reference_positions_nm[None])[0][0]

    def support_mask(self, x: Array) -> Array:
        return self.coordinates.support_mask(x)

    def source(self) -> Molecular_Source:
        ratio = self.temperature_kelvin / self.bundle_temperature_kelvin
        return Molecular_Source(
            self.domain,
            self.source_mean,
            self.source_variance * ratio,
        )

    def regularized(self, rg_param) -> "Regularized_Molecular_Potential":
        """The ``rho = (e, r)`` surrogate: energy threshold ``e`` [kJ/mol], pair floor ``r`` [nm]."""
        return Regularized_Molecular_Potential(self, rg_param)


class Regularized_Molecular_Potential(Potential):
    """The two-parameter regularized potential ``U^{rho}`` of the manuscript.

    With ``rho = (e, r)``: the regular and exception nonbonded pair distances
    are floored at ``r`` (nm), the excess ``Delta E`` of the floored energy over
    the floored energy of the bundle reference geometry is compressed by
    ``C_e(Delta E) = Delta E`` below ``e`` and ``e (1 + log(Delta E / e))``
    above it (kJ/mol); the reduced potential is ``beta (E_star + C_e) - log J``.
    The physical energy stays unregularized. ``e -> inf, r -> 0`` recovers
    the base potential.
    """

    base: Molecular_Potential
    rg_param: Array
    reference_energy_kj_mol: Array

    def __init__(self, base: Molecular_Potential, rg_param):
        self.base = base
        self.rg_param = jnp.asarray(rg_param, dtype=base.reference_positions_nm.dtype)
        self.reference_energy_kj_mol = base.forcefield.energy_with_pair_distance_floor(
            base.reference_positions_nm[None], self.rg_param[1]
        )[0]

    @property
    def domain(self):
        return self.base.domain

    @property
    def dimension(self) -> int:
        return self.base.dimension

    @property
    def beta(self) -> Array:
        return self.base.beta

    @property
    def energy_threshold_kj_mol(self) -> Array:
        return self.rg_param[0]

    @property
    def pair_distance_floor_nm(self) -> Array:
        return self.rg_param[1]

    def _compress(self, energy: Array) -> Array:
        excess = energy - self.reference_energy_kj_mol
        threshold = self.rg_param[0]
        active = excess > threshold
        safe = jnp.where(active, excess, threshold)
        mapped = threshold * (1.0 + jnp.log(safe / threshold))
        return self.reference_energy_kj_mol + jnp.where(active, mapped, excess)

    def regularized_energy(self, q: Array) -> Array:
        """``E_star + C_e(Delta E_r)`` in kJ/mol."""
        x = self.base.coordinates.to_cartesian(q)[0]
        return self._compress(
            self.base.forcefield.energy_with_pair_distance_floor(x, self.rg_param[1])
        )

    def reduced_energy(self, q: Array) -> Array:
        return self.beta * self.regularized_energy(q)

    def __call__(self, q: Array) -> Array:
        x, logdet = self.base.coordinates.to_cartesian(q)
        energy = self._compress(
            self.base.forcefield.energy_with_pair_distance_floor(x, self.rg_param[1])
        )
        return self.beta * energy - logdet

    def physical_energy(self, q: Array) -> Array:
        return self.base.physical_energy(q)

    def cartesian(self, q: Array) -> Array:
        return self.base.cartesian(q)

    def reference_internal(self) -> Array:
        return self.base.reference_internal()

    def support_mask(self, x: Array) -> Array:
        return self.base.support_mask(x)

    def source(self) -> Molecular_Source:
        return self.base.source()
