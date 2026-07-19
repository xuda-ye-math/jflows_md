"""Pure-JAX Amber bonded/nonbonded plus OBC1/ACE energy."""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
from jax import Array


COULOMB = 138.9354576
GB_COULOMB = 138.935485
GB_OFFSET_NM = 0.009


def _distance_with_zero_subgradient(distance_squared: Array) -> Array:
    """Take a square root with the declared zero subgradient at coincidence."""

    positive = distance_squared > 0
    safe_squared = jnp.where(positive, distance_squared, 1.0)
    return jnp.where(positive, jnp.sqrt(safe_squared), 0.0)


def _array(
    spec: Mapping,
    name: str,
    *,
    integer: bool = False,
    columns: int | None = None,
) -> Array:
    dtype = jnp.int32 if integer else None
    value = jnp.asarray(spec[name], dtype=dtype)
    if columns is not None:
        value = value.reshape((-1, columns))
    return value


class Amber_OBC_Force_Field(eqx.Module):
    """Static-array reimplementation of an audited OpenMM Amber/OBC1 System."""

    n_atoms: int = eqx.field(static=True)
    bond_idx: Array
    bond_length: Array
    bond_k: Array
    angle_idx: Array
    angle_theta: Array
    angle_k: Array
    torsion_idx: Array
    torsion_periodicity: Array
    torsion_phase: Array
    torsion_k: Array
    pair_idx: Array
    pair_chargeprod: Array
    pair_sigma: Array
    pair_epsilon: Array
    exception_idx: Array
    exception_chargeprod: Array
    exception_sigma: Array
    exception_epsilon: Array
    gb_charge: Array
    gb_or: Array
    gb_sr: Array
    solute_dielectric: Array
    solvent_dielectric: Array
    ace_coefficient: Array

    def __init__(self, spec: Mapping):
        self.n_atoms = int(spec["n_atoms"])
        self.bond_idx = _array(spec, "bond_idx", integer=True, columns=2)
        self.bond_length = _array(spec, "bond_length_nm")
        self.bond_k = _array(spec, "bond_k_kj_mol_nm2")
        self.angle_idx = _array(spec, "angle_idx", integer=True, columns=3)
        self.angle_theta = _array(spec, "angle_theta_rad")
        self.angle_k = _array(spec, "angle_k_kj_mol_rad2")
        self.torsion_idx = _array(spec, "torsion_idx", integer=True, columns=4)
        self.torsion_periodicity = _array(spec, "torsion_periodicity")
        self.torsion_phase = _array(spec, "torsion_phase_rad")
        self.torsion_k = _array(spec, "torsion_k_kj_mol")
        self.pair_idx = _array(spec, "pair_idx", integer=True, columns=2)
        self.pair_chargeprod = _array(spec, "pair_chargeprod_e2")
        self.pair_sigma = _array(spec, "pair_sigma_nm")
        self.pair_epsilon = _array(spec, "pair_epsilon_kj_mol")
        self.exception_idx = _array(
            spec, "exception_idx", integer=True, columns=2
        )
        self.exception_chargeprod = _array(spec, "exception_chargeprod_e2")
        self.exception_sigma = _array(spec, "exception_sigma_nm")
        self.exception_epsilon = _array(spec, "exception_epsilon_kj_mol")
        self.gb_charge = _array(spec, "gb_charge_e")
        self.gb_or = _array(spec, "gb_offset_radius_nm")
        self.gb_sr = _array(spec, "gb_scaled_offset_radius_nm")
        self.solute_dielectric = jnp.asarray(spec["solute_dielectric"])
        self.solvent_dielectric = jnp.asarray(spec["solvent_dielectric"])
        self.ace_coefficient = jnp.asarray(spec["ace_coefficient_kj_mol_nm2"])

    @staticmethod
    def _dihedral(x: Array, idx: Array) -> Array:
        p0, p1 = x[:, idx[:, 0]], x[:, idx[:, 1]]
        p2, p3 = x[:, idx[:, 2]], x[:, idx[:, 3]]
        b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
        n1, n2 = jnp.cross(b1, b2), jnp.cross(b2, b3)
        b2_hat = b2 / jnp.linalg.norm(b2, axis=-1, keepdims=True)
        return jnp.arctan2(
            jnp.sum(jnp.cross(n1, n2) * b2_hat, axis=-1),
            jnp.sum(n1 * n2, axis=-1),
        )

    @staticmethod
    def _pair_energy(
        x: Array, idx: Array, chargeprod: Array, sigma: Array, epsilon: Array
    ) -> Array:
        distance = jnp.linalg.norm(x[:, idx[:, 0]] - x[:, idx[:, 1]], axis=-1)
        inverse = 1.0 / distance
        sr6 = (sigma * inverse) ** 6
        return jnp.sum(
            COULOMB * chargeprod * inverse + 4.0 * epsilon * (sr6 * sr6 - sr6),
            axis=-1,
        )

    @staticmethod
    def _pair_energy_with_floor(
        x: Array,
        idx: Array,
        chargeprod: Array,
        sigma: Array,
        epsilon: Array,
        pair_distance_floor_nm: Array,
    ) -> Array:
        """Return nonbonded energy at a hard pair-distance floor."""

        displacement = x[:, idx[:, 0]] - x[:, idx[:, 1]]
        distance = _distance_with_zero_subgradient(
            jnp.sum(displacement * displacement, axis=-1)
        )
        inverse = 1.0 / jnp.maximum(distance, pair_distance_floor_nm)
        sr6 = (sigma * inverse) ** 6
        return jnp.sum(
            COULOMB * chargeprod * inverse
            + 4.0 * epsilon * (sr6 * sr6 - sr6),
            axis=-1,
        )

    @staticmethod
    def _obc_descreening_integral(
        distance: Array,
        radius_i: Array,
        scaled_j: Array,
        eye: Array,
    ) -> Array:
        """Stable Amber ``igb=2`` directed descreening integral.

        OpenMM's exact expression is retained away from coincidence. Small
        ``d/s`` series avoid subtracting divergent terms for ``s >= a``.
        The limiting value and chosen Cartesian subgradient at ``d=0`` are
        both zero; the latter is supplied by ``_distance_with_zero_subgradient``.
        """

        dtype = distance.dtype
        series_ratio = jnp.asarray(jnp.finfo(dtype).eps ** 0.25, dtype=dtype)
        safe_scaled = jnp.where(scaled_j > 0, scaled_j, 1.0)
        scaled_ratio = distance / safe_scaled
        small = scaled_ratio < series_ratio
        series_gt = (
            scaled_ratio
            + scaled_ratio**2 / 3.0
            + scaled_ratio**3
            + 2.0 * scaled_ratio**4 / 5.0
            + scaled_ratio**5
            + 3.0 * scaled_ratio**6 / 7.0
        ) / safe_scaled
        use_series_gt = (
            (~eye)
            & (scaled_j > radius_i)
            & (distance < scaled_j - radius_i)
            & small
        )

        safe_radius = jnp.where(radius_i > 0, radius_i, 1.0)
        radius_ratio = distance / safe_radius
        series_eq = (
            radius_ratio / 4.0
            - radius_ratio**2 / 3.0
            + 5.0 * radius_ratio**3 / 16.0
            - 3.0 * radius_ratio**4 / 10.0
            + 7.0 * radius_ratio**5 / 24.0
            - 2.0 * radius_ratio**6 / 7.0
        ) / safe_radius
        use_series_eq = (
            (~eye) & (scaled_j == radius_i) & (radius_ratio < series_ratio)
        )

        use_general = (~eye) & (~use_series_gt) & (~use_series_eq)
        safe_distance = jnp.where(use_general & (distance > 0), distance, 1.0)
        upper = safe_distance + scaled_j
        lower = jnp.maximum(radius_i, jnp.abs(safe_distance - scaled_j))
        safe_upper = jnp.where(use_general, upper, 1.0)
        safe_lower = jnp.where(use_general, lower, 1.0)
        inv_lower, inv_upper = 1.0 / safe_lower, 1.0 / safe_upper
        general = 0.5 * (
            inv_lower
            - inv_upper
            + 0.25
            * (safe_distance - scaled_j * scaled_j / safe_distance)
            * (inv_upper * inv_upper - inv_lower * inv_lower)
            + 0.5 * jnp.log(safe_lower / safe_upper) / safe_distance
        )
        integral = jnp.where(
            use_series_gt,
            series_gt,
            jnp.where(use_series_eq, series_eq, general),
        )
        valid = (distance + scaled_j - radius_i >= 0.0) & (~eye)
        return jnp.where(valid, integral, 0.0)

    def _gb_energy(self, x: Array) -> Array:
        difference = x[:, :, None, :] - x[:, None, :, :]
        distance_squared = jnp.sum(difference * difference, axis=-1)
        eye = jnp.eye(self.n_atoms, dtype=bool)[None, :, :]
        integral_distance = _distance_with_zero_subgradient(distance_squared)
        radius_i = self.gb_or[None, :, None]
        scaled_j = self.gb_sr[None, None, :]
        integral = self._obc_descreening_integral(
            integral_distance,
            radius_i,
            scaled_j,
            eye,
        )
        born_integral = jnp.sum(integral, axis=2)
        psi = born_integral * self.gb_or[None, :]
        full_radius = self.gb_or + GB_OFFSET_NM
        tanh_argument = 0.8 * psi + 2.909125 * psi**3
        born = 1.0 / (
            1.0 / self.gb_or[None, :]
            - jnp.tanh(tanh_argument) / full_radius[None, :]
        )

        born_pair = born[:, :, None] * born[:, None, :]
        f = jnp.sqrt(
            distance_squared
            + born_pair * jnp.exp(-distance_squared / (4.0 * born_pair))
        )
        charge_pair = self.gb_charge[None, :, None] * self.gb_charge[None, None, :]
        prefactor = -0.5 * GB_COULOMB * (
            1.0 / self.solute_dielectric - 1.0 / self.solvent_dielectric
        )
        polarization = prefactor * jnp.sum(charge_pair / f, axis=(1, 2))
        ace = self.ace_coefficient * jnp.sum(
            (full_radius[None, :] + 0.14) ** 2
            * (full_radius[None, :] / born) ** 6,
            axis=1,
        )
        return polarization + ace

    def _bonded_and_gb_terms(self, x: Array) -> dict[str, Array]:
        """Evaluate terms shared by the raw and floor-aware energy paths."""

        bond_distance = jnp.linalg.norm(
            x[:, self.bond_idx[:, 0]] - x[:, self.bond_idx[:, 1]], axis=-1
        )
        bond = jnp.sum(
            0.5 * self.bond_k * (bond_distance - self.bond_length) ** 2, axis=-1
        )
        first = x[:, self.angle_idx[:, 0]] - x[:, self.angle_idx[:, 1]]
        second = x[:, self.angle_idx[:, 2]] - x[:, self.angle_idx[:, 1]]
        cosine = jnp.sum(first * second, axis=-1) / (
            jnp.linalg.norm(first, axis=-1) * jnp.linalg.norm(second, axis=-1)
        )
        theta = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
        angle = jnp.sum(
            0.5 * self.angle_k * (theta - self.angle_theta) ** 2, axis=-1
        )
        phi = self._dihedral(x, self.torsion_idx)
        torsion = jnp.sum(
            self.torsion_k
            * (1.0 + jnp.cos(self.torsion_periodicity * phi - self.torsion_phase)),
            axis=-1,
        )
        return {
            "bond": bond,
            "angle": angle,
            "torsion": torsion,
            "gb": self._gb_energy(x),
        }

    def energy_terms(self, x: Array) -> dict[str, Array]:
        terms = self._bonded_and_gb_terms(x)
        nonbonded = self._pair_energy(
            x, self.pair_idx, self.pair_chargeprod, self.pair_sigma, self.pair_epsilon
        ) + self._pair_energy(
            x,
            self.exception_idx,
            self.exception_chargeprod,
            self.exception_sigma,
            self.exception_epsilon,
        )
        total = (
            terms["bond"]
            + terms["angle"]
            + terms["torsion"]
            + nonbonded
            + terms["gb"]
        )
        return {
            "bond": terms["bond"],
            "angle": terms["angle"],
            "torsion": terms["torsion"],
            "nonbonded": nonbonded,
            "gb": terms["gb"],
            "total": total,
        }

    def _energy_with_pair_distance_floor(
        self, x: Array, pair_distance_floor_nm: Array
    ) -> Array:
        """Return total energy with only nonbonded pair distances floored."""

        terms = self._bonded_and_gb_terms(x)
        nonbonded = self._pair_energy_with_floor(
            x,
            self.pair_idx,
            self.pair_chargeprod,
            self.pair_sigma,
            self.pair_epsilon,
            pair_distance_floor_nm,
        ) + self._pair_energy_with_floor(
            x,
            self.exception_idx,
            self.exception_chargeprod,
            self.exception_sigma,
            self.exception_epsilon,
            pair_distance_floor_nm,
        )
        return (
            terms["bond"]
            + terms["angle"]
            + terms["torsion"]
            + nonbonded
            + terms["gb"]
        )

    def __call__(self, x: Array) -> Array:
        return self.energy_terms(x)["total"]
