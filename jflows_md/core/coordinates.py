"""Differentiable mixed-domain BAT coordinates with exact Jacobians."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import operator

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .domain import Mixed_Domain


def _logit(value: Array) -> Array:
    return jnp.log(value) - jnp.log1p(-value)


def signed_volume(
    x: Array, center: int, first: int, second: int, third: int
) -> Array:
    c = x[..., center, :]
    a = x[..., first, :] - c
    b = x[..., second, :] - c
    d = x[..., third, :] - c
    return jnp.sum(a * jnp.cross(b, d), axis=-1)


def _integer(value, *, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return result


def _atom_quadruple(value, *, name: str) -> tuple[int, int, int, int]:
    try:
        atoms = tuple(
            _integer(atom, name=f"{name}[{position}]")
            for position, atom in enumerate(value)
        )
    except TypeError as exc:
        raise ValueError(f"{name} must contain four atom indices") from exc
    if len(atoms) != 4:
        raise ValueError(f"{name} must contain four atom indices")
    return atoms


def _normalize_fixed_stereocenters(
    spec: Mapping,
) -> tuple[tuple[str, int, int, tuple[int, int, int, int], int], ...]:
    """Normalize the schema-2 singleton and schema-3 explicit list."""

    legacy_keys = {
        "chiral_torsion_index",
        "chiral_torsion_sign",
        "chirality_atoms",
        "chirality_sign",
    }
    if "fixed_stereocenters" in spec:
        if legacy_keys & spec.keys():
            raise ValueError(
                "coordinate spec cannot mix fixed_stereocenters with legacy "
                "singleton chirality fields"
            )
        raw = spec["fixed_stereocenters"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError("fixed_stereocenters must be a sequence")
        result = []
        for position, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"fixed_stereocenters[{position}] must be a mapping"
                )
            label = str(item.get("label", f"stereocenter_{position}"))
            if not label:
                raise ValueError(
                    f"fixed_stereocenters[{position}].label must be nonempty"
            )
            try:
                torsion_index = _integer(
                    item["torsion_index"],
                    name=f"fixed_stereocenters[{position}].torsion_index",
                )
                torsion_sign = _integer(
                    item["torsion_sign"],
                    name=f"fixed_stereocenters[{position}].torsion_sign",
                )
                volume_sign = _integer(
                    item["volume_sign"],
                    name=f"fixed_stereocenters[{position}].volume_sign",
                )
                atoms = _atom_quadruple(
                    item["atoms"],
                    name=f"fixed_stereocenters[{position}].atoms",
                )
            except KeyError as exc:
                raise ValueError(
                    f"fixed_stereocenters[{position}] is missing {exc.args[0]!r}"
                ) from exc
            result.append(
                (label, torsion_index, torsion_sign, atoms, volume_sign)
            )
        return tuple(result)

    torsion_index = _integer(
        spec.get("chiral_torsion_index", -1), name="chiral_torsion_index"
    )
    torsion_sign = _integer(
        spec.get("chiral_torsion_sign", 0), name="chiral_torsion_sign"
    )
    atoms = _atom_quadruple(
        spec.get("chirality_atoms", (-1, -1, -1, -1)),
        name="chirality_atoms",
    )
    volume_sign = _integer(
        spec.get("chirality_sign", 0), name="chirality_sign"
    )
    if torsion_index < 0:
        if torsion_sign != 0 or volume_sign != 0:
            raise ValueError(
                "unconstrained legacy coordinates require zero chirality signs"
            )
        return ()
    return (("legacy", torsion_index, torsion_sign, atoms, volume_sign),)


class Internal_Coordinates(eqx.Module):
    """Almost-everywhere BAT chart on ``R^p x T^q``.

    Bonds use offset/scaled log coordinates, angles use offset/scaled logits
    of ``angle/pi``, and ordinary torsions remain periodic. An optional chiral
    torsion is replaced by ``tau = sign*pi*sigmoid(eta)``. Any number of
    explicitly configured torsions may define fixed stereochemical
    half-charts. Coordinate specs use the standard Cartesian configurational
    measure after quotienting rigid translations and rotations.
    """

    order: tuple[int, ...] = eqx.field(static=True)
    refs: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    n_atoms: int = eqx.field(static=True)
    n_bonds: int = eqx.field(static=True)
    n_angles: int = eqx.field(static=True)
    n_torsions: int = eqx.field(static=True)
    jacobian_measure: str = eqx.field(static=True)
    fixed_stereocenter_labels: tuple[str, ...] = eqx.field(static=True)
    stereocenter_torsion_indices: tuple[int, ...] = eqx.field(static=True)
    stereocenter_torsion_signs: tuple[int, ...] = eqx.field(static=True)
    stereocenter_atoms: tuple[tuple[int, int, int, int], ...] = eqx.field(
        static=True
    )
    stereocenter_volume_signs: tuple[int, ...] = eqx.field(static=True)
    n_fixed_stereocenters: int = eqx.field(static=True)
    # Schema-2 singleton aliases are retained for existing diagnostics.
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
        schema_version = _integer(
            spec.get("schema_version", 2), name="schema_version"
        )
        if schema_version not in (2, 3):
            raise ValueError(
                f"unsupported coordinate schema_version: {schema_version}"
            )
        expected_chart = (
            f"log-bond_logit-angle_quotient_BAT_v{schema_version}"
        )
        if spec.get("chart") != expected_chart:
            raise ValueError(
                f"coordinate chart must be {expected_chart!r}, "
                f"got {spec.get('chart')!r}"
            )
        self.order = tuple(
            _integer(atom, name=f"order[{position}]")
            for position, atom in enumerate(spec["order"])
        )
        self.refs = tuple(
            tuple(
                _integer(reference, name=f"refs[{row}][{column}]")
                for column, reference in enumerate(values)
            )
            for row, values in enumerate(spec["refs"])
        )
        self.n_atoms = len(self.order)
        if self.n_atoms < 3 or sorted(self.order) != list(range(self.n_atoms)):
            raise ValueError("coordinate order must be a permutation of atom indices")
        if len(self.refs) != self.n_atoms or any(
            len(row) != 3 for row in self.refs
        ):
            raise ValueError("coordinate refs must contain one triple per atom")
        placed = set()
        for placement, (atom, row) in enumerate(
            zip(self.order, self.refs, strict=True)
        ):
            required = row[: min(placement, 3)]
            if len(set(required)) != len(required) or any(
                reference not in placed for reference in required
            ):
                raise ValueError(
                    f"invalid coordinate refs at placement {placement}: {row}"
                )
            placed.add(atom)
        self.n_bonds = self.n_atoms - 1
        self.n_angles = self.n_atoms - 2
        self.n_torsions = self.n_atoms - 3
        self.jacobian_measure = str(spec.get("jacobian_measure"))
        if self.jacobian_measure != "rigid_motion_quotient_v1":
            raise ValueError(
                f"unsupported coordinate Jacobian measure: {self.jacobian_measure}"
            )

        fixed = _normalize_fixed_stereocenters(spec)
        labels = tuple(item[0] for item in fixed)
        torsion_indices = tuple(item[1] for item in fixed)
        torsion_signs = tuple(item[2] for item in fixed)
        atoms = tuple(item[3] for item in fixed)
        volume_signs = tuple(item[4] for item in fixed)
        if len(set(labels)) != len(labels):
            raise ValueError("fixed stereocenter labels must be unique")
        if len(set(torsion_indices)) != len(torsion_indices):
            raise ValueError("fixed stereocenter torsion indices must be unique")
        if len({item[0] for item in atoms}) != len(atoms):
            raise ValueError("fixed stereocenter center atoms must be unique")
        if any(
            index < 0 or index >= self.n_torsions
            for index in torsion_indices
        ):
            raise ValueError("fixed stereocenter torsion index is out of range")
        if any(sign not in (-1, 1) for sign in torsion_signs):
            raise ValueError("fixed stereocenter torsion signs must be -1 or 1")
        if any(sign not in (-1, 1) for sign in volume_signs):
            raise ValueError("fixed stereocenter volume signs must be -1 or 1")
        for position, (torsion_index, center_atoms) in enumerate(
            zip(torsion_indices, atoms, strict=True)
        ):
            if len(set(center_atoms)) != 4 or any(
                atom < 0 or atom >= self.n_atoms for atom in center_atoms
            ):
                raise ValueError(
                    f"fixed_stereocenters[{position}].atoms must contain four "
                    "distinct in-range indices"
                )
            placement = torsion_index + 3
            placed_atom = self.order[placement]
            r1, r2, r3 = self.refs[placement]
            if center_atoms[0] != r1 or set(center_atoms[1:]) != {
                r2,
                r3,
                placed_atom,
            }:
                raise ValueError(
                    f"fixed_stereocenters[{position}] is not represented by "
                    "its selected Z-matrix torsion"
                )

        self.fixed_stereocenter_labels = labels
        self.stereocenter_torsion_indices = torsion_indices
        self.stereocenter_torsion_signs = torsion_signs
        self.stereocenter_atoms = atoms
        self.stereocenter_volume_signs = volume_signs
        self.n_fixed_stereocenters = len(fixed)
        if self.n_fixed_stereocenters == 1:
            self.chiral_torsion_index = torsion_indices[0]
            self.chiral_torsion_sign = torsion_signs[0]
            self.chirality_atoms = atoms[0]
            self.chirality_sign = volume_signs[0]
        else:
            self.chiral_torsion_index = -1
            self.chiral_torsion_sign = 0
            self.chirality_atoms = (-1, -1, -1, -1)
            self.chirality_sign = 0

        expected_periodic = self.n_torsions - self.n_fixed_stereocenters
        expected_euclidean = (
            self.n_bonds + self.n_angles + self.n_fixed_stereocenters
        )
        expected_dimension = 3 * self.n_atoms - 6
        for name, expected in (
            ("dimension", expected_dimension),
            ("euclidean_dim", expected_euclidean),
            ("periodic_dim", expected_periodic),
        ):
            if name in spec and _integer(spec[name], name=name) != expected:
                raise ValueError(
                    f"coordinate {name} must be {expected}, got {spec[name]}"
                )
        self.domain = Mixed_Domain(expected_euclidean, expected_periodic)

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
            [
                index
                for index in range(self.n_torsions)
                if index not in self.stereocenter_torsion_indices
            ],
            dtype=jnp.int32,
        )
        self.bond_offset = jnp.asarray(spec["bond_log_offset"])
        self.bond_scale = jnp.asarray(spec["bond_log_scale"])
        self.angle_offset = jnp.asarray(spec["angle_logit_offset"])
        self.angle_scale = jnp.asarray(spec["angle_logit_scale"])
        for name, value, expected in (
            ("bond_log_offset", self.bond_offset, self.n_bonds),
            ("bond_log_scale", self.bond_scale, self.n_bonds),
            ("angle_logit_offset", self.angle_offset, self.n_angles),
            ("angle_logit_scale", self.angle_scale, self.n_angles),
        ):
            if value.shape != (expected,):
                raise ValueError(
                    f"coordinate {name} must have shape {(expected,)}, "
                    f"got {value.shape}"
                )
        if not bool(jnp.all(jnp.isfinite(self.bond_offset))):
            raise ValueError("bond_log_offset must be finite")
        if not bool(
            jnp.all(jnp.isfinite(self.bond_scale) & (self.bond_scale > 0.0))
        ):
            raise ValueError("bond_log_scale must be finite and strictly positive")
        if not bool(jnp.all(jnp.isfinite(self.angle_offset))):
            raise ValueError("angle_logit_offset must be finite")
        if not bool(
            jnp.all(jnp.isfinite(self.angle_scale) & (self.angle_scale > 0.0))
        ):
            raise ValueError("angle_logit_scale must be finite and strictly positive")

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

    def _decode(
        self, q: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        bond_q = q[:, : self.n_bonds]
        angle_q = q[:, self.n_bonds : self.n_bonds + self.n_angles]
        bond_log = self.bond_offset + self.bond_scale * bond_q
        angle_logit = self.angle_offset + self.angle_scale * angle_q
        bonds = jnp.exp(bond_log)
        epsilon = jnp.sqrt(jnp.finfo(q.dtype).eps)
        angle_fraction = jnp.clip(
            jax.nn.sigmoid(angle_logit), epsilon, 1.0 - epsilon
        )
        angles = jnp.pi * angle_fraction
        periodic = q[:, self.domain.euclidean_dim :]
        if self.n_fixed_stereocenters == 0:
            torsions = periodic
            chiral_fraction = jnp.empty((q.shape[0], 0), dtype=q.dtype)
            chiral_eta = jnp.empty((q.shape[0], 0), dtype=q.dtype)
        else:
            start = self.n_bonds + self.n_angles
            chiral_eta = q[:, start : start + self.n_fixed_stereocenters]
            chiral_fraction = jnp.clip(
                jax.nn.sigmoid(chiral_eta), epsilon, 1.0 - epsilon
            )
            pieces = []
            ordinary = 0
            for index in range(self.n_torsions):
                if index in self.stereocenter_torsion_indices:
                    position = self.stereocenter_torsion_indices.index(index)
                    pieces.append(
                        self.stereocenter_torsion_signs[position]
                        * jnp.pi
                        * chiral_fraction[:, position]
                    )
                else:
                    pieces.append(periodic[:, ordinary])
                    ordinary += 1
            torsions = jnp.stack(pieces, axis=-1)
        return bonds, angles, torsions, chiral_fraction, angle_logit, chiral_eta

    def _logdet(self, bonds: Array, angle_logit: Array, chiral_eta: Array) -> Array:
        # sin(theta) is symmetric about pi/2. Evaluating it through the smaller
        # boundary distance avoids float32 sigmoid saturation producing
        # sin(pi) < 0 and hence NaNs at the open chart boundary.
        boundary_angle = jnp.pi * jax.nn.sigmoid(-jnp.abs(angle_logit))
        log_sin = jnp.log(jnp.sin(boundary_angle))
        # Standard Z-matrix volume element after factoring the six rigid
        # degrees of freedom: prod_i r_i^2 prod_j sin(theta_j).
        bat = jnp.sum(2.0 * jnp.log(bonds), axis=-1)
        bat = bat + jnp.sum(log_sin, axis=-1)
        chart = jnp.sum(jnp.log(self.bond_scale) + jnp.log(bonds), axis=-1)
        chart = chart + jnp.sum(
            jnp.log(self.angle_scale)
            + jnp.log(jnp.pi)
            + jax.nn.log_sigmoid(angle_logit)
            + jax.nn.log_sigmoid(-angle_logit),
            axis=-1,
        )
        if self.n_fixed_stereocenters == 1:
            eta = chiral_eta[:, 0]
            chart = (
                chart
                + jnp.log(jnp.pi)
                + jax.nn.log_sigmoid(eta)
                + jax.nn.log_sigmoid(-eta)
            )
        elif self.n_fixed_stereocenters > 1:
            chart = chart + jnp.sum(
                jnp.log(jnp.pi)
                + jax.nn.log_sigmoid(chiral_eta)
                + jax.nn.log_sigmoid(-chiral_eta),
                axis=-1,
            )
        return bat + chart

    def to_internal(self, x: Array) -> tuple[Array, Array]:
        """Canonical Cartesian representatives to ``q`` and negative log volume."""

        bonds, angles, torsions = self.raw_internal(x)
        bond_q = (jnp.log(bonds) - self.bond_offset) / self.bond_scale
        epsilon = jnp.sqrt(jnp.finfo(x.dtype).eps)
        fraction = jnp.clip(angles / jnp.pi, epsilon, 1.0 - epsilon)
        angle_q = (_logit(fraction) - self.angle_offset) / self.angle_scale
        angle_logit = self.angle_offset + self.angle_scale * angle_q
        if self.n_fixed_stereocenters == 0:
            q = jnp.concatenate((bond_q, angle_q, torsions), axis=-1)
            chiral_eta = jnp.empty((x.shape[0], 0), dtype=x.dtype)
        elif self.n_fixed_stereocenters == 1:
            tau = torsions[:, self.stereocenter_torsion_indices[0]]
            chiral_fraction = jnp.clip(
                self.stereocenter_torsion_signs[0] * tau / jnp.pi,
                epsilon,
                1.0 - epsilon,
            )
            eta = _logit(chiral_fraction)
            ordinary = torsions[:, self.ordinary_torsion_indices]
            q = jnp.concatenate(
                (bond_q, angle_q, eta[:, None], ordinary), axis=-1
            )
            chiral_eta = eta[:, None]
        else:
            tau = jnp.stack(
                [
                    torsions[:, index]
                    for index in self.stereocenter_torsion_indices
                ],
                axis=-1,
            )
            signs = jnp.asarray(
                self.stereocenter_torsion_signs, dtype=x.dtype
            )
            chiral_fraction = jnp.clip(
                signs * tau / jnp.pi,
                epsilon,
                1.0 - epsilon,
            )
            chiral_eta = _logit(chiral_fraction)
            ordinary = torsions[:, self.ordinary_torsion_indices]
            q = jnp.concatenate(
                (bond_q, angle_q, chiral_eta, ordinary), axis=-1
            )
        logdet = self._logdet(bonds, angle_logit, chiral_eta)
        return q, -logdet

    def to_cartesian(self, q: Array) -> tuple[Array, Array]:
        """Return canonical positions and the configured configurational log volume."""

        bonds, angles, torsions, _, angle_logit, chiral_eta = self._decode(q)
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
        return x, self._logdet(bonds, angle_logit, chiral_eta)

    def support_mask(self, x: Array) -> Array:
        if self.n_fixed_stereocenters == 0:
            return jnp.ones(x.shape[:-2], dtype=bool)
        checks = []
        for atoms, sign in zip(
            self.stereocenter_atoms,
            self.stereocenter_volume_signs,
            strict=True,
        ):
            center, first, second, third = atoms
            volume = signed_volume(x, center, first, second, third)
            checks.append(sign * volume > 0.0)
        return jnp.all(jnp.stack(checks, axis=-1), axis=-1)
