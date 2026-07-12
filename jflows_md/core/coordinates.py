"""Differentiable mixed-domain BAT coordinates with exact Jacobians."""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .chirality import signed_volume, support_sign
from .domain import Mixed_Domain


def _logit(value: Array) -> Array:
    return jnp.log(value) - jnp.log1p(-value)


class Internal_Coordinates(eqx.Module):
    """Almost-everywhere BAT chart on ``R^p x T^q``.

    Bonds use offset/scaled log coordinates, angles use offset/scaled logits
    of ``angle/pi``, and ordinary torsions remain periodic. An optional chiral
    torsion is replaced by ``tau = sign*pi*sigmoid(eta)``.
    """

    order: tuple[int, ...] = eqx.field(static=True)
    refs: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    n_atoms: int = eqx.field(static=True)
    n_bonds: int = eqx.field(static=True)
    n_angles: int = eqx.field(static=True)
    n_torsions: int = eqx.field(static=True)
    chiral_torsion_index: int = eqx.field(static=True)
    chiral_torsion_sign: int = eqx.field(static=True)
    chirality_atoms: tuple[int, int, int, int] = eqx.field(static=True)
    chirality_sign: int = eqx.field(static=True)
    domain: Mixed_Domain
    bond_atom: Array
    bond_ref: Array
    angle_atom: Array
    angle_ref1: Array
    angle_ref2: Array
    torsion_atom: Array
    torsion_ref1: Array
    torsion_ref2: Array
    torsion_ref3: Array
    ordinary_torsion_indices: Array
    bond_offset: Array
    bond_scale: Array
    angle_offset: Array
    angle_scale: Array

    def __init__(self, spec: Mapping):
        self.order = tuple(map(int, spec["order"]))
        self.refs = tuple(tuple(map(int, row)) for row in spec["refs"])
        self.n_atoms = len(self.order)
        self.n_bonds = self.n_atoms - 1
        self.n_angles = self.n_atoms - 2
        self.n_torsions = self.n_atoms - 3
        self.chiral_torsion_index = int(spec.get("chiral_torsion_index", -1))
        self.chiral_torsion_sign = int(spec.get("chiral_torsion_sign", 0))
        self.chirality_atoms = tuple(map(int, spec.get("chirality_atoms", (-1, -1, -1, -1))))
        self.chirality_sign = int(spec.get("chirality_sign", 0))
        expected_periodic = self.n_torsions - int(self.chiral_torsion_index >= 0)
        expected_euclidean = self.n_bonds + self.n_angles + int(self.chiral_torsion_index >= 0)
        self.domain = Mixed_Domain(expected_euclidean, expected_periodic)
        if int(spec["euclidean_dim"]) != expected_euclidean:
            raise ValueError("CoordinateSpec Euclidean dimension is inconsistent")
        if int(spec["periodic_dim"]) != expected_periodic:
            raise ValueError("CoordinateSpec periodic dimension is inconsistent")

        order = self.order
        refs = self.refs
        self.bond_atom = jnp.asarray(order[1:], dtype=jnp.int32)
        self.bond_ref = jnp.asarray([refs[p][0] for p in range(1, self.n_atoms)], dtype=jnp.int32)
        self.angle_atom = jnp.asarray(order[2:], dtype=jnp.int32)
        self.angle_ref1 = jnp.asarray([refs[p][0] for p in range(2, self.n_atoms)], dtype=jnp.int32)
        self.angle_ref2 = jnp.asarray([refs[p][1] for p in range(2, self.n_atoms)], dtype=jnp.int32)
        self.torsion_atom = jnp.asarray(order[3:], dtype=jnp.int32)
        self.torsion_ref1 = jnp.asarray([refs[p][0] for p in range(3, self.n_atoms)], dtype=jnp.int32)
        self.torsion_ref2 = jnp.asarray([refs[p][1] for p in range(3, self.n_atoms)], dtype=jnp.int32)
        self.torsion_ref3 = jnp.asarray([refs[p][2] for p in range(3, self.n_atoms)], dtype=jnp.int32)
        self.ordinary_torsion_indices = jnp.asarray(
            [i for i in range(self.n_torsions) if i != self.chiral_torsion_index],
            dtype=jnp.int32,
        )
        self.bond_offset = jnp.asarray(spec["bond_log_offset"])
        self.bond_scale = jnp.asarray(spec["bond_log_scale"])
        self.angle_offset = jnp.asarray(spec["angle_logit_offset"])
        self.angle_scale = jnp.asarray(spec["angle_logit_scale"])
        if self.bond_offset.shape != (self.n_bonds,) or self.bond_scale.shape != (self.n_bonds,):
            raise ValueError("invalid bond coordinate arrays")
        if self.angle_offset.shape != (self.n_angles,) or self.angle_scale.shape != (self.n_angles,):
            raise ValueError("invalid angle coordinate arrays")
        if bool(jnp.any(self.bond_scale <= 0)) or bool(jnp.any(self.angle_scale <= 0)):
            raise ValueError("coordinate scales must be positive")
        if self.chiral_torsion_index >= 0:
            if self.chiral_torsion_sign not in (-1, 1) or self.chirality_sign not in (-1, 1):
                raise ValueError("chiral CoordinateSpec requires signed support metadata")

    @staticmethod
    def _dihedral4(a: Array, b: Array, c: Array, d: Array) -> Array:
        b1, b2, b3 = b - a, c - b, d - c
        n1 = jnp.cross(b1, b2)
        n2 = jnp.cross(b2, b3)
        b2_hat = b2 / jnp.linalg.norm(b2, axis=-1, keepdims=True)
        m1 = jnp.cross(n1, b2_hat)
        return jnp.arctan2(jnp.sum(m1 * n2, axis=-1), jnp.sum(n1 * n2, axis=-1))

    def raw_internal(self, x: Array) -> tuple[Array, Array, Array]:
        bonds = jnp.linalg.norm(x[:, self.bond_atom] - x[:, self.bond_ref], axis=-1)
        v1 = x[:, self.angle_ref2] - x[:, self.angle_ref1]
        v2 = x[:, self.angle_atom] - x[:, self.angle_ref1]
        cosine = jnp.sum(v1 * v2, axis=-1) / (
            jnp.linalg.norm(v1, axis=-1) * jnp.linalg.norm(v2, axis=-1)
        )
        angles = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
        torsions = self._dihedral4(
            x[:, self.torsion_ref3],
            x[:, self.torsion_ref2],
            x[:, self.torsion_ref1],
            x[:, self.torsion_atom],
        )
        return bonds, angles, torsions

    def _decode(self, q: Array) -> tuple[Array, Array, Array, Array]:
        bond_q = q[:, : self.n_bonds]
        angle_q = q[:, self.n_bonds : self.n_bonds + self.n_angles]
        bond_log = self.bond_offset + self.bond_scale * bond_q
        angle_logit = self.angle_offset + self.angle_scale * angle_q
        bonds = jnp.exp(bond_log)
        angle_fraction = jax.nn.sigmoid(angle_logit)
        angles = jnp.pi * angle_fraction
        periodic = q[:, self.domain.euclidean_dim :]
        if self.chiral_torsion_index < 0:
            torsions = periodic
            chiral_fraction = jnp.empty((q.shape[0], 0), dtype=q.dtype)
        else:
            eta = q[:, self.n_bonds + self.n_angles]
            chiral_fraction = jax.nn.sigmoid(eta)[:, None]
            pieces = []
            ordinary = 0
            for index in range(self.n_torsions):
                if index == self.chiral_torsion_index:
                    pieces.append(self.chiral_torsion_sign * jnp.pi * chiral_fraction[:, 0])
                else:
                    pieces.append(periodic[:, ordinary])
                    ordinary += 1
            torsions = jnp.stack(pieces, axis=-1)
        return bonds, angles, torsions, chiral_fraction

    def _logdet(self, bonds: Array, angles: Array, chiral_fraction: Array) -> Array:
        bat = jnp.log(bonds[:, 1])
        bat = bat + jnp.sum(
            2.0 * jnp.log(bonds[:, 2:]) + jnp.log(jnp.sin(angles[:, 1:])), axis=-1
        )
        chart = jnp.sum(jnp.log(self.bond_scale) + jnp.log(bonds), axis=-1)
        angle_fraction = angles / jnp.pi
        chart = chart + jnp.sum(
            jnp.log(self.angle_scale)
            + jnp.log(jnp.pi)
            + jnp.log(angle_fraction)
            + jnp.log1p(-angle_fraction),
            axis=-1,
        )
        if self.chiral_torsion_index >= 0:
            fraction = chiral_fraction[:, 0]
            chart = chart + jnp.log(jnp.pi) + jnp.log(fraction) + jnp.log1p(-fraction)
        return bat + chart

    def to_internal(self, x: Array) -> tuple[Array, Array]:
        """Cartesian ``[batch, atoms, 3]`` to chart coordinates and log|dq/dx|."""

        bonds, angles, torsions = self.raw_internal(x)
        bond_q = (jnp.log(bonds) - self.bond_offset) / self.bond_scale
        fraction = jnp.clip(angles / jnp.pi, 1e-14, 1.0 - 1e-14)
        angle_q = (_logit(fraction) - self.angle_offset) / self.angle_scale
        if self.chiral_torsion_index < 0:
            q = jnp.concatenate((bond_q, angle_q, torsions), axis=-1)
            chiral_fraction = jnp.empty((x.shape[0], 0), dtype=x.dtype)
        else:
            tau = torsions[:, self.chiral_torsion_index]
            chiral_fraction = jnp.clip(
                self.chiral_torsion_sign * tau / jnp.pi, 1e-14, 1.0 - 1e-14
            )
            eta = _logit(chiral_fraction)
            ordinary = torsions[:, self.ordinary_torsion_indices]
            q = jnp.concatenate((bond_q, angle_q, eta[:, None], ordinary), axis=-1)
            chiral_fraction = chiral_fraction[:, None]
        logdet = self._logdet(bonds, angles, chiral_fraction)
        return q, -logdet

    def to_cartesian(self, q: Array) -> tuple[Array, Array]:
        """Chart coordinates to canonical-frame Cartesian positions and log|dx/dq|."""

        bonds, angles, torsions, chiral_fraction = self._decode(q)
        batch = q.shape[0]
        positions: list[Array | None] = [None] * self.n_atoms
        root = self.order
        positions[root[0]] = jnp.zeros((batch, 3), dtype=q.dtype)
        positions[root[1]] = positions[root[0]] + jnp.stack(
            (bonds[:, 0], jnp.zeros(batch, q.dtype), jnp.zeros(batch, q.dtype)), axis=-1
        )

        r1, r2, _ = self.refs[2]
        vector = positions[r2] - positions[r1]
        unit = vector / jnp.linalg.norm(vector, axis=-1, keepdims=True)
        perpendicular = jnp.stack(
            (-unit[:, 1], unit[:, 0], jnp.zeros(batch, q.dtype)), axis=-1
        )
        positions[root[2]] = positions[r1] + bonds[:, 1, None] * (
            jnp.cos(angles[:, 0, None]) * unit
            + jnp.sin(angles[:, 0, None]) * perpendicular
        )

        for placement in range(3, self.n_atoms):
            atom = root[placement]
            r1, r2, r3 = self.refs[placement]
            c, b, a = positions[r1], positions[r2], positions[r3]
            bc = c - b
            bc = bc / jnp.linalg.norm(bc, axis=-1, keepdims=True)
            normal = jnp.cross(b - a, bc)
            normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
            m = jnp.cross(normal, bc)
            distance = bonds[:, placement - 1]
            angle = angles[:, placement - 2]
            torsion = torsions[:, placement - 3]
            offset = (
                -(distance * jnp.cos(angle))[:, None] * bc
                + (distance * jnp.sin(angle) * jnp.cos(torsion))[:, None] * m
                - (distance * jnp.sin(angle) * jnp.sin(torsion))[:, None] * normal
            )
            positions[atom] = c + offset

        x = jnp.stack(positions, axis=1)
        return x, self._logdet(bonds, angles, chiral_fraction)

    def support_mask(self, x: Array) -> Array:
        if self.chirality_sign == 0:
            return jnp.ones(x.shape[:-2], dtype=bool)
        center, first, second, third = self.chirality_atoms
        volume = signed_volume(x, center, first, second, third)
        return support_sign(volume, self.chirality_sign)
