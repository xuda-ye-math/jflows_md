"""Bundle-driven molecular potential on mixed internal coordinates."""

from __future__ import annotations

import hashlib
import math
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


__all__ = ["KB_KJ_MOL_K", "Molecular_Potential"]


class Molecular_Potential(Potential):
    """Physical molecular target pulled back to a bundle-frozen BAT chart.

    ``q`` has shape ``[batch, d]`` and the returned reduced energy is

    ``beta * E_bundle(x(q)) - log J_config(q)``.

    Current schema-2 targets use the standard Cartesian configurational
    measure with global translation and rotation factored out. The canonical
    Cartesian frame returned by :meth:`cartesian` is a representative, not a
    set of six physical holonomic constraints. An explicitly supplied legacy
    schema-1 bundle retains its historical gauge-slice measure.
    """

    forcefield: Amber_OBC_Force_Field
    coordinates: Internal_Coordinates
    reference_positions_nm: Array
    beta: Array
    temperature_kelvin: float = eqx.field(static=True)
    bundle_name: str = eqx.field(static=True)
    bundle_path: str = eqx.field(static=True)
    manifest_sha256: str = eqx.field(static=True)

    def __init__(self, bundle: Molecular_Bundle):
        self.forcefield = Amber_OBC_Force_Field(bundle.system)
        self.coordinates = Internal_Coordinates(bundle.coordinates)
        self.reference_positions_nm = jnp.asarray(bundle.validation["frames_nm"][0])
        self.temperature_kelvin = float(bundle.manifest["temperature_kelvin"])
        if not math.isfinite(self.temperature_kelvin) or self.temperature_kelvin <= 0:
            raise ValueError("bundle temperature must be positive and finite")
        self.beta = jnp.asarray(1.0 / (KB_KJ_MOL_K * self.temperature_kelvin))
        self.bundle_name = bundle.name
        self.bundle_path = str(bundle.path)
        self.manifest_sha256 = hashlib.sha256(
            (bundle.path / "manifest.json").read_bytes()
        ).hexdigest()
        if self.forcefield.n_atoms != self.coordinates.n_atoms:
            raise ValueError("SystemSpec and CoordinateSpec atom counts differ")
        if self.reference_positions_nm.shape != (self.forcefield.n_atoms, 3):
            raise ValueError("bundle reference positions have the wrong shape")
        if not bool(jnp.all(jnp.isfinite(self.reference_positions_nm))):
            raise ValueError("bundle reference positions must be finite")

    @classmethod
    def from_bundle(
        cls, path_or_name: str | Path, *, verify: bool = True
    ) -> "Molecular_Potential":
        return cls(Molecular_Bundle.load(path_or_name, verify=verify))

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

    def source(self) -> Molecular_Source:
        bundle = Molecular_Bundle.load(self.bundle_path, verify=True)
        return Molecular_Source.from_spec(self.domain, bundle.coordinates)

    def regularized(
        self,
        energy_cut_kj_mol: float,
        *,
        energy_scale_kj_mol: float = 50.0,
        tail_fraction: float = 0.0,
    ) -> Potential:
        """Return an explicit soft-energy surrogate of this exact target.

        The physical Cartesian energy is shifted by the bundle reference
        energy and left unchanged up to ``energy_cut_kj_mol``. Above that
        excess-energy cutoff, the remainder is compressed by a C1 lin-log
        map. ``tail_fraction`` optionally retains a linear asymptotic tail;
        zero reproduces the archived molecular soft-cap shape but requires a
        target-specific normalizability audit. A positive value preserves a
        coercive fraction of the physical tail. The complete coordinate
        Jacobian remains exact and unregularized.

        This method never mutates the physical target. The returned potential
        is a deliberately deformed diagnostic/training target and must be
        sharpened back to ``self`` before physical evaluation.
        """

        return _Regularized_Molecular_Potential(
            self,
            energy_cut_kj_mol=energy_cut_kj_mol,
            energy_scale_kj_mol=energy_scale_kj_mol,
            tail_fraction=tail_fraction,
        )


class _Regularized_Molecular_Potential(Potential):
    """Private implementation returned by :meth:`Molecular_Potential.regularized`."""

    base: Molecular_Potential
    reference_energy_kj_mol: Array
    energy_cut_kj_mol: Array
    energy_scale_kj_mol: Array
    tail_fraction: Array

    def __init__(
        self,
        base: Molecular_Potential,
        *,
        energy_cut_kj_mol: float,
        energy_scale_kj_mol: float,
        tail_fraction: float,
    ):
        try:
            energy_cut_kj_mol = float(energy_cut_kj_mol)
            energy_scale_kj_mol = float(energy_scale_kj_mol)
            tail_fraction = float(tail_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("molecular regularization values must be real scalars") from exc
        if not math.isfinite(energy_cut_kj_mol) or energy_cut_kj_mol <= 0:
            raise ValueError("energy_cut_kj_mol must be positive and finite")
        if not math.isfinite(energy_scale_kj_mol) or energy_scale_kj_mol <= 0:
            raise ValueError("energy_scale_kj_mol must be positive and finite")
        if not math.isfinite(tail_fraction) or not 0.0 <= tail_fraction <= 1.0:
            raise ValueError("tail_fraction must lie in [0, 1]")
        self.base = base
        reference = base.forcefield(base.reference_positions_nm[None, ...])[0]
        self.reference_energy_kj_mol = jnp.asarray(reference)
        dtype = self.reference_energy_kj_mol.dtype
        self.energy_cut_kj_mol = jnp.asarray(energy_cut_kj_mol, dtype=dtype)
        self.energy_scale_kj_mol = jnp.asarray(energy_scale_kj_mol, dtype=dtype)
        self.tail_fraction = jnp.asarray(tail_fraction, dtype=dtype)

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
    def manifest_sha256(self) -> str:
        return self.base.manifest_sha256

    def cartesian(self, q: Array) -> Array:
        return self.base.cartesian(q)

    def physical_energy(self, q: Array) -> Array:
        return self.base.physical_energy(q)

    def _regularize_energy(self, energy: Array) -> Array:
        excess = energy - self.reference_energy_kj_mol
        over = jnp.maximum(excess - self.energy_cut_kj_mol, 0.0)
        log_tail = self.energy_scale_kj_mol * jnp.log1p(
            over / self.energy_scale_kj_mol
        )
        log_fraction = 1.0 - self.tail_fraction
        compressed_log = jnp.where(
            log_fraction == 0.0,
            jnp.zeros_like(log_tail),
            log_fraction * log_tail,
        )
        compressed_linear = jnp.where(
            self.tail_fraction == 0.0,
            jnp.zeros_like(over),
            self.tail_fraction * over,
        )
        compressed = (
            self.energy_cut_kj_mol
            + compressed_log
            + compressed_linear
        )
        regularized_excess = jnp.where(
            excess > self.energy_cut_kj_mol,
            compressed,
            excess,
        )
        return self.reference_energy_kj_mol + regularized_excess

    def regularized_physical_energy(self, q: Array) -> Array:
        """Return the deformed Cartesian energy in kJ/mol."""

        return self._regularize_energy(self.base.physical_energy(q))

    def __call__(self, q: Array) -> Array:
        self.base._validate_internal(q)
        x, logdet = self.base.coordinates.to_cartesian(q)
        energy = self._regularize_energy(self.base.forcefield(x))
        return self.base.beta * energy - logdet

    def reference_internal(self) -> Array:
        return self.base.reference_internal()

    def support_mask(self, x: Array) -> Array:
        return self.base.support_mask(x)

    def source(self) -> Molecular_Source:
        return self.base.source()
