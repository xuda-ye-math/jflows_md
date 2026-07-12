"""Bundle-driven molecular potential on mixed internal coordinates."""

from __future__ import annotations

import hashlib
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

    ``beta * E_bundle(x(q)) - log|dx/dq|``.
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
