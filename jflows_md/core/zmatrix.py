"""Deterministic Z-matrix construction from a molecular bond graph."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


Reference = tuple[int, int, int]


def _best_candidate(
    candidates: Iterable[int],
    *,
    positions: np.ndarray | None,
    score,
) -> int:
    values = sorted(set(int(x) for x in candidates))
    if not values:
        raise ValueError("no valid Z-matrix reference candidate")
    if positions is None:
        return values[0]
    return max(values, key=lambda value: (float(score(value)), -value))


def build_zmatrix(
    bonds: Sequence[Sequence[int]],
    n_atoms: int,
    *,
    positions: np.ndarray | None = None,
    root: int | None = None,
    prefix: Sequence[int] = (),
    overrides: Mapping[int, Reference] | None = None,
) -> tuple[tuple[int, ...], tuple[Reference, ...]]:
    """Return deterministic placement order and references.

    ``overrides`` maps an atom index to its ``(bond, angle, torsion)``
    references. Every referenced atom must already occur in the order.
    ``prefix`` is useful when a stereochemical torsion must occupy a known BAT
    slot, as for the ADP CB half-chart.
    """

    if n_atoms < 4:
        raise ValueError("molecular BAT coordinates require at least four atoms")
    adj = [set() for _ in range(n_atoms)]
    for pair in bonds:
        if len(pair) != 2:
            raise ValueError(f"invalid bond: {pair}")
        a, b = map(int, pair)
        if a == b or not (0 <= a < n_atoms and 0 <= b < n_atoms):
            raise ValueError(f"invalid bond indices: {pair}")
        adj[a].add(b)
        adj[b].add(a)
    if any(not neighbors for neighbors in adj):
        raise ValueError("bond graph contains an isolated atom")

    prefix = tuple(map(int, prefix))
    if len(set(prefix)) != len(prefix):
        raise ValueError("Z-matrix prefix contains duplicate atoms")
    if any(not (0 <= atom < n_atoms) for atom in prefix):
        raise ValueError("Z-matrix prefix atom is out of range")
    if root is None:
        root = prefix[0] if prefix else max(range(n_atoms), key=lambda i: (len(adj[i]), -i))
    if prefix and prefix[0] != root:
        raise ValueError("prefix[0] must equal the requested root")

    order = list(prefix or (root,))
    seen = set(order)
    queue = deque(order)
    while queue:
        atom = queue.popleft()
        for neighbor in sorted(adj[atom]):
            if neighbor not in seen:
                seen.add(neighbor)
                order.append(neighbor)
                queue.append(neighbor)
    if len(order) != n_atoms:
        raise ValueError("bond graph is disconnected")

    pos = None if positions is None else np.asarray(positions, dtype=float)
    if pos is not None and pos.shape != (n_atoms, 3):
        raise ValueError(f"positions must have shape {(n_atoms, 3)}, got {pos.shape}")
    overrides = dict(overrides or {})
    refs: list[Reference] = []
    placed: list[int] = []
    placed_set: set[int] = set()

    for placement, atom in enumerate(order):
        if atom in overrides:
            ref = tuple(map(int, overrides[atom]))
            needed = ref[: min(placement, 3)]
            if any(value not in placed_set for value in needed):
                raise ValueError(f"override for atom {atom} references an unplaced atom: {ref}")
            if placement >= 1 and ref[0] not in adj[atom]:
                raise ValueError(f"override bond reference is not bonded for atom {atom}: {ref}")
            refs.append(ref)
            placed.append(atom)
            placed_set.add(atom)
            continue

        if placement == 0:
            ref = (-1, -1, -1)
        else:
            bonded = adj[atom] & placed_set
            if not bonded:
                raise ValueError(f"atom {atom} has no already placed bonded reference")
            r1 = _best_candidate(
                bonded,
                positions=pos,
                score=lambda candidate: -placed.index(candidate),
            )
            if placement == 1:
                ref = (r1, -1, -1)
            else:
                r2_candidates = (adj[r1] & placed_set) - {atom}
                if not r2_candidates:
                    r2_candidates = placed_set - {r1}
                r2 = _best_candidate(
                    r2_candidates,
                    positions=pos,
                    score=lambda candidate: np.linalg.norm(
                        np.cross(pos[atom] - pos[r1], pos[candidate] - pos[r1])
                    ),
                )
                if placement == 2:
                    ref = (r1, r2, -1)
                else:
                    r3_candidates = (adj[r2] & placed_set) - {atom, r1}
                    if not r3_candidates:
                        r3_candidates = placed_set - {r1, r2}
                    r3 = _best_candidate(
                        r3_candidates,
                        positions=pos,
                        score=lambda candidate: np.linalg.norm(
                            np.cross(pos[r1] - pos[r2], pos[candidate] - pos[r2])
                        ),
                    )
                    ref = (r1, r2, r3)
        refs.append(ref)
        placed.append(atom)
        placed_set.add(atom)

    return tuple(order), tuple(refs)


def validate_zmatrix(
    order: Sequence[int], refs: Sequence[Sequence[int]], bonds: Sequence[Sequence[int]]
) -> None:
    n_atoms = len(order)
    if sorted(order) != list(range(n_atoms)) or len(refs) != n_atoms:
        raise ValueError("invalid Z-matrix order or reference count")
    bond_set = {frozenset(map(int, pair)) for pair in bonds}
    placed: set[int] = set()
    for placement, (atom, ref) in enumerate(zip(order, refs, strict=True)):
        required = tuple(ref)[: min(placement, 3)]
        if len(set(required)) != len(required) or any(value not in placed for value in required):
            raise ValueError(f"invalid references at placement {placement}: {ref}")
        if placement and frozenset((int(atom), int(ref[0]))) not in bond_set:
            raise ValueError(f"bond reference is not bonded at placement {placement}: {ref}")
        placed.add(int(atom))
