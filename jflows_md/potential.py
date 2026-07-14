"""Bundle-driven molecular potential on mixed internal coordinates."""

from __future__ import annotations

import math
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from jflows.potential import Potential

from .core.coordinates import Internal_Coordinates
from .core.forcefield import COULOMB, Amber_OBC_Force_Field
from .source import Molecular_Source
from .system import Molecular_Bundle


KB_KJ_MOL_K = 0.00831446261815324


__all__ = ["KB_KJ_MOL_K", "Molecular_Potential"]


class Molecular_Potential(Potential):
    """Physical molecular target pulled back to a bundle-frozen BAT chart.

    ``q`` has shape ``[batch, d]`` and the returned reduced energy is

    ``beta * E_bundle(x(q)) - log J_config(q)``.

    Targets use the standard Cartesian configurational measure with global
    translation and rotation factored out. The canonical Cartesian frame
    returned by :meth:`cartesian` is a representative, not a set of six
    physical holonomic constraints.
    """

    forcefield: Amber_OBC_Force_Field
    coordinates: Internal_Coordinates
    reference_positions_nm: Array
    beta: Array
    temperature_kelvin: float = eqx.field(static=True)
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
        if temperature_kelvin is None:
            temperature_kelvin = bundle.manifest["temperature_kelvin"]
        self.temperature_kelvin = float(temperature_kelvin)
        if not math.isfinite(self.temperature_kelvin) or self.temperature_kelvin <= 0:
            raise ValueError("bundle temperature must be positive and finite")
        self.beta = jnp.asarray(1.0 / (KB_KJ_MOL_K * self.temperature_kelvin))
        self.bundle_name = bundle.name
        self.bundle_path = str(bundle.path)
        if self.forcefield.n_atoms != self.coordinates.n_atoms:
            raise ValueError("SystemSpec and CoordinateSpec atom counts differ")
        if self.reference_positions_nm.shape != (self.forcefield.n_atoms, 3):
            raise ValueError("bundle reference positions have the wrong shape")
        if not bool(jnp.all(jnp.isfinite(self.reference_positions_nm))):
            raise ValueError("bundle reference positions must be finite")

    @classmethod
    def from_bundle(
        cls,
        path_or_name: str | Path,
        *,
        verify: bool = True,
        temperature_kelvin: float | None = None,
    ) -> "Molecular_Potential":
        """Load bundle mechanics, optionally overriding its target temperature.

        The override changes the inverse-temperature factor multiplying the
        bundle's physical energy. :meth:`source` also scales the bundle
        Gaussian's Euclidean variance by ``temperature / bundle_temperature``
        so source and target temperatures remain matched. It does not mutate
        the bundle or alter its force field, solvent model, coordinates,
        periodic-uniform source component, or reference geometry.
        """

        return cls(
            Molecular_Bundle.load(path_or_name, verify=verify),
            temperature_kelvin=temperature_kelvin,
        )

    @property
    def domain(self):
        return self.coordinates.domain

    @property
    def dimension(self) -> int:
        return self.domain.dimension

    def _validate_internal(self, q: Array) -> None:
        self.domain._validate(q, "molecular internal coordinates")
        if q.ndim != 2:
            raise ValueError(
                f"molecular internal coordinates must have shape [N, d], got {q.shape}"
            )

    def cartesian(self, q: Array) -> Array:
        self._validate_internal(q)
        return self.coordinates.to_cartesian(q)[0]

    def physical_energy(self, q: Array) -> Array:
        self._validate_internal(q)
        return self.forcefield(self.cartesian(q))

    def energy_terms(self, q: Array) -> dict[str, Array]:
        self._validate_internal(q)
        x, logdet = self.coordinates.to_cartesian(q)
        terms = self.forcefield.energy_terms(x)
        reduced = {name: self.beta * value for name, value in terms.items()}
        reduced["logdet"] = logdet
        reduced["potential"] = reduced["total"] - logdet
        return reduced

    def __call__(self, q: Array) -> Array:
        self._validate_internal(q)
        x, logdet = self.coordinates.to_cartesian(q)
        return self.beta * self.forcefield(x) - logdet

    def reference_internal(self) -> Array:
        return self.coordinates.to_internal(self.reference_positions_nm[None])[0][0]

    def support_mask(self, x: Array) -> Array:
        return self.coordinates.support_mask(x)

    def source(
        self,
        *,
        defensive_weight: float = 0.0,
        defensive_df: float = 3.0,
    ) -> Molecular_Source:
        """Return the source matched to this target's temperature.

        The bundle stores a Gaussian reference at its manifest temperature.
        Its Euclidean variance follows the harmonic scaling ``variance ~ T``;
        the mean and uniform periodic component are temperature independent.
        At the bundle temperature this is exactly the archived source.
        """

        bundle = Molecular_Bundle.load(self.bundle_path, verify=True)
        source = Molecular_Source.from_spec(self.domain, bundle.coordinates)
        bundle_temperature = float(bundle.manifest["temperature_kelvin"])
        temperature_ratio = self.temperature_kelvin / bundle_temperature
        return Molecular_Source(
            self.domain,
            mean=source.mean,
            variance=source.variance * temperature_ratio,
            defensive_weight=defensive_weight,
            defensive_df=defensive_df,
        )

    def regularized(
        self,
        energy_threshold_kj_mol: float,
        *,
        pair_distance_floor_nm: float = 0.0,
    ) -> Potential:
        """Return an immutable energy/distance-regularized surrogate.

        ``energy_threshold_kj_mol`` is the positive, reference-relative
        threshold of a C1 linear/logarithmic energy map. A positive
        ``pair_distance_floor_nm`` additionally floors distances used by
        Amber regular-pair and exception Coulomb/Lennard-Jones terms. OBC1,
        bonded terms, and the coordinate Jacobian remain unfloored.
        Positive floors that are too small to keep the active nonbonded terms
        finite in the force-field dtype are rejected eagerly; zero remains the
        explicit energy-only mode.

        This method never mutates the physical target. The returned potential
        is a deliberately deformed training/diagnostic bridge and is not an
        automatically certified normalizable or physical endpoint. Sharpen
        to ``self`` before physical evaluation. The dimensionless training
        option ``e_clip`` is independent of this kJ/mol threshold.
        """

        return _Regularized_Molecular_Potential(
            self,
            energy_threshold_kj_mol=energy_threshold_kj_mol,
            pair_distance_floor_nm=pair_distance_floor_nm,
        )


class _Regularized_Molecular_Potential(Potential):
    """Private implementation returned by :meth:`Molecular_Potential.regularized`."""

    base: Molecular_Potential
    reference_energy_kj_mol: Array
    energy_threshold_kj_mol: Array
    pair_distance_floor_nm: Array

    def __init__(
        self,
        base: Molecular_Potential,
        *,
        energy_threshold_kj_mol: float,
        pair_distance_floor_nm: float,
    ):
        energy_threshold_kj_mol = _regularization_scalar(
            "energy_threshold_kj_mol", energy_threshold_kj_mol
        )
        pair_distance_floor_nm = _regularization_scalar(
            "pair_distance_floor_nm", pair_distance_floor_nm
        )
        if energy_threshold_kj_mol <= 0:
            raise ValueError("energy_threshold_kj_mol must be positive")
        if pair_distance_floor_nm < 0:
            raise ValueError("pair_distance_floor_nm must be nonnegative")
        self.base = base
        dtype = base.reference_positions_nm.dtype
        with np.errstate(over="ignore", under="ignore"):
            threshold_host = np.asarray(
                energy_threshold_kj_mol, dtype=np.dtype(dtype)
            )
            floor_host = np.asarray(pair_distance_floor_nm, dtype=np.dtype(dtype))
        threshold = jnp.asarray(threshold_host)
        floor = jnp.asarray(floor_host)
        cast_threshold = float(threshold)
        cast_floor = float(floor)
        if not math.isfinite(cast_threshold) or cast_threshold <= 0:
            raise ValueError(
                "energy_threshold_kj_mol must remain positive and finite "
                f"after casting to {dtype}"
            )
        if not math.isfinite(cast_floor) or cast_floor < 0:
            raise ValueError(
                "pair_distance_floor_nm must remain nonnegative and finite "
                f"after casting to {dtype}"
            )
        if pair_distance_floor_nm > 0 and cast_floor == 0:
            raise ValueError(
                "pair_distance_floor_nm underflows to zero in the force-field dtype"
            )
        minimum_floor = _minimum_finite_pair_floor_nm(base.forcefield, dtype)
        if 0 < cast_floor < minimum_floor:
            raise ValueError(
                "pair_distance_floor_nm is too small for finite nonbonded "
                f"evaluation in {dtype}; require 0 or at least "
                f"{minimum_floor:.8g} nm"
            )
        self.energy_threshold_kj_mol = threshold
        self.pair_distance_floor_nm = floor
        reference = base.forcefield._energy_with_pair_distance_floor(
            base.reference_positions_nm[None, ...], floor
        )[0]
        self.reference_energy_kj_mol = jnp.asarray(reference, dtype=dtype)

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
    def temperature_kelvin(self) -> float:
        return self.base.temperature_kelvin

    @property
    def bundle_name(self) -> str:
        return self.base.bundle_name

    @property
    def bundle_path(self) -> str:
        return self.base.bundle_path

    def cartesian(self, q: Array) -> Array:
        return self.base.cartesian(q)

    def physical_energy(self, q: Array) -> Array:
        return self.base.physical_energy(q)

    def _regularize_energy(self, energy: Array) -> Array:
        excess = energy - self.reference_energy_kj_mol
        threshold = self.energy_threshold_kj_mol
        active = excess > threshold
        safe_over = jnp.where(active, excess - threshold, threshold)
        log_ratio = (
            jnp.logaddexp(jnp.log(safe_over), jnp.log(threshold))
            - jnp.log(threshold)
        )
        compressed = threshold + threshold * log_ratio
        regularized_excess = jnp.where(
            active,
            compressed,
            excess,
        )
        return self.reference_energy_kj_mol + regularized_excess

    def regularized_energy(self, q: Array) -> Array:
        """Return the floor-aware, energy-regularized surrogate in kJ/mol."""

        self.base._validate_internal(q)
        x = self.base.cartesian(q)
        floor_energy = self.base.forcefield._energy_with_pair_distance_floor(
            x, self.pair_distance_floor_nm
        )
        return self._regularize_energy(floor_energy)

    def __call__(self, q: Array) -> Array:
        self.base._validate_internal(q)
        x, logdet = self.base.coordinates.to_cartesian(q)
        floor_energy = self.base.forcefield._energy_with_pair_distance_floor(
            x, self.pair_distance_floor_nm
        )
        energy = self._regularize_energy(floor_energy)
        return self.base.beta * energy - logdet

    def reference_internal(self) -> Array:
        return self.base.reference_internal()

    def support_mask(self, x: Array) -> Array:
        return self.base.support_mask(x)

    def source(self, **kwargs) -> Molecular_Source:
        return self.base.source(**kwargs)


def _regularization_scalar(name: str, value) -> float:
    """Validate one finite, non-Boolean, scalar host regularizer."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real scalar, not a boolean")
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a real scalar") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be a real scalar, not a boolean")
    try:
        result = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _minimum_finite_pair_floor_nm(
    forcefield: Amber_OBC_Force_Field, dtype
) -> float:
    """Conservative positive floor for finite dtype-level pair arithmetic."""

    sigma = np.concatenate(
        (
            np.asarray(forcefield.pair_sigma, dtype=np.float64),
            np.asarray(forcefield.exception_sigma, dtype=np.float64),
        )
    )
    epsilon = np.concatenate(
        (
            np.asarray(forcefield.pair_epsilon, dtype=np.float64),
            np.asarray(forcefield.exception_epsilon, dtype=np.float64),
        )
    )
    chargeprod = np.concatenate(
        (
            np.asarray(forcefield.pair_chargeprod, dtype=np.float64),
            np.asarray(forcefield.exception_chargeprod, dtype=np.float64),
        )
    )
    interaction_count = max(int(sigma.size), 1)
    maximum = float(np.finfo(np.dtype(dtype)).max)
    minimum = 2.0 / maximum if sigma.size else 0.0

    active_lj = (sigma > 0) & (epsilon > 0)
    if np.any(active_lj):
        active_sigma = sigma[active_lj]
        active_epsilon = epsilon[active_lj]
        energy_bound = np.exp(
            np.log(active_sigma)
            + (
                np.log(16.0 * interaction_count)
                + np.log(active_epsilon)
                - np.log(maximum)
            )
            / 12.0
        )
        intermediate_bound = np.exp(
            np.log(active_sigma)
            + (np.log(2.0) - np.log(maximum)) / 12.0
        )
        minimum = max(
            minimum,
            float(np.max(energy_bound)),
            float(np.max(intermediate_bound)),
        )

    active_charge = chargeprod != 0
    if np.any(active_charge):
        coulomb_bound = (
            4.0
            * interaction_count
            * COULOMB
            * np.abs(chargeprod[active_charge])
            / maximum
        )
        minimum = max(minimum, float(np.max(coulomb_bound)))
    return minimum
