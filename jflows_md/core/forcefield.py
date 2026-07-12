"""Pure-JAX Amber bonded/nonbonded plus OBC1/ACE energy."""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
from jax import Array


COULOMB = 138.9354576
GB_COULOMB = 138.935485
GB_OFFSET_NM = 0.009


def _array(spec: Mapping, name: str, *, integer: bool = False) -> Array:
    dtype = jnp.int32 if integer else None
    return jnp.asarray(spec[name], dtype=dtype)


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
        self.bond_idx = _array(spec, "bond_idx", integer=True)
        self.bond_length = _array(spec, "bond_length_nm")
        self.bond_k = _array(spec, "bond_k_kj_mol_nm2")
        self.angle_idx = _array(spec, "angle_idx", integer=True)
        self.angle_theta = _array(spec, "angle_theta_rad")
        self.angle_k = _array(spec, "angle_k_kj_mol_rad2")
        self.torsion_idx = _array(spec, "torsion_idx", integer=True)
        self.torsion_periodicity = _array(spec, "torsion_periodicity")
        self.torsion_phase = _array(spec, "torsion_phase_rad")
        self.torsion_k = _array(spec, "torsion_k_kj_mol")
        self.pair_idx = _array(spec, "pair_idx", integer=True)
        self.pair_chargeprod = _array(spec, "pair_chargeprod_e2")
        self.pair_sigma = _array(spec, "pair_sigma_nm")
        self.pair_epsilon = _array(spec, "pair_epsilon_kj_mol")
        self.exception_idx = _array(spec, "exception_idx", integer=True)
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
        if idx.shape[0] == 0:
            return jnp.zeros(x.shape[0], dtype=x.dtype)
        distance = jnp.linalg.norm(x[:, idx[:, 0]] - x[:, idx[:, 1]], axis=-1)
        inverse = 1.0 / distance
        sr6 = (sigma * inverse) ** 6
        return jnp.sum(
            COULOMB * chargeprod * inverse + 4.0 * epsilon * (sr6 * sr6 - sr6),
            axis=-1,
        )

    def _gb_energy(self, x: Array) -> Array:
        difference = x[:, :, None, :] - x[:, None, :, :]
        distance_squared = jnp.sum(difference * difference, axis=-1)
        eye = jnp.eye(self.n_atoms, dtype=bool)[None, :, :]
        # The diagonal has r=0 physically, but differentiating norm(0) produces
        # NaNs. The descreening diagonal is excluded, so give only that branch a
        # constant safe distance. Polarization below uses r^2 directly and
        # retains the exact self term f_ii=B_i with a well-defined derivative.
        integral_distance = jnp.sqrt(jnp.where(eye, 1.0, distance_squared))
        radius_i = self.gb_or[None, :, None]
        scaled_j = self.gb_sr[None, None, :]
        upper = integral_distance + scaled_j
        lower = jnp.maximum(radius_i, jnp.abs(integral_distance - scaled_j))
        inv_lower, inv_upper = 1.0 / lower, 1.0 / upper
        integral = 0.5 * (
            inv_lower
            - inv_upper
            + 0.25
            * (integral_distance - scaled_j * scaled_j / integral_distance)
            * (inv_upper * inv_upper - inv_lower * inv_lower)
            + 0.5 * jnp.log(lower / upper) / integral_distance
        )
        valid = (integral_distance + scaled_j - radius_i >= 0.0) & (~eye)
        born_integral = jnp.sum(jnp.where(valid, integral, 0.0), axis=2)
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

    def energy_terms(self, x: Array) -> dict[str, Array]:
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
        angle = jnp.sum(0.5 * self.angle_k * (theta - self.angle_theta) ** 2, axis=-1)
        phi = self._dihedral(x, self.torsion_idx)
        torsion = jnp.sum(
            self.torsion_k
            * (1.0 + jnp.cos(self.torsion_periodicity * phi - self.torsion_phase)),
            axis=-1,
        )
        nonbonded = self._pair_energy(
            x, self.pair_idx, self.pair_chargeprod, self.pair_sigma, self.pair_epsilon
        ) + self._pair_energy(
            x,
            self.exception_idx,
            self.exception_chargeprod,
            self.exception_sigma,
            self.exception_epsilon,
        )
        gb = self._gb_energy(x)
        total = bond + angle + torsion + nonbonded + gb
        return {
            "bond": bond,
            "angle": angle,
            "torsion": torsion,
            "nonbonded": nonbonded,
            "gb": gb,
            "total": total,
        }

    def __call__(self, x: Array) -> Array:
        return self.energy_terms(x)["total"]
