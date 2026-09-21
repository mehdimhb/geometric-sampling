"""Reversible atom-block moves projected exactly onto discrete counts."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from .constraints import MomentConstraints


@dataclass(frozen=True)
class DiscreteCountState:
    """A validated count macrostate for a fixed set of moment constraints."""

    counts: np.ndarray
    constraints: MomentConstraints

    def __init__(
        self,
        counts: Sequence[int],
        constraints: MomentConstraints,
    ) -> None:
        validated = constraints.validate_counts(counts).copy()
        validated.setflags(write=False)
        object.__setattr__(self, "counts", validated)
        object.__setattr__(self, "constraints", constraints)

    @property
    def D(self) -> int:
        return int(self.counts.sum())

    @property
    def probabilities(self) -> np.ndarray:
        return self.counts.astype(float) / self.D

    def copy(self) -> "DiscreteCountState":
        return DiscreteCountState(self.counts.copy(), self.constraints)


@dataclass(frozen=True)
class StepResult:
    """Summary of one accepted microscopic proposal (possibly a self-loop)."""

    block_size: int
    source_labels: tuple[int, ...]
    target_labels: tuple[int, ...]
    changed: bool


@dataclass(frozen=True)
class ChainResult:
    """Final state and checkpoint trace from a count-projected atom chain."""

    final_counts: np.ndarray
    checkpoint_iterations: np.ndarray
    checkpoint_counts: np.ndarray
    proposals: int
    changes: int
    burn_in: int
    runtime_seconds: float
    proposals_by_block: dict[int, int]
    changes_by_block: dict[int, int]
    initial_atoms: np.ndarray | None = None
    final_atoms: np.ndarray | None = None

    @property
    def change_rate(self) -> float:
        return self.changes / self.proposals if self.proposals else 0.0


class ReplacementKernel:
    """State-independent mixture of uniform within-signature replacements."""

    def __init__(
        self,
        constraints: MomentConstraints,
        block_weights: Mapping[int, float],
        *,
        max_patterns: int = 1_000_000,
    ) -> None:
        if not block_weights:
            raise ValueError("block_weights must not be empty")
        if not isinstance(max_patterns, (int, np.integer)) or max_patterns <= 0:
            raise ValueError("max_patterns must be a positive integer")

        cleaned: dict[int, float] = {}
        for raw_k, raw_weight in block_weights.items():
            if not isinstance(raw_k, (int, np.integer)) or int(raw_k) <= 0:
                raise ValueError("block sizes must be positive integers")
            weight = float(raw_weight)
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("block weights must be finite and nonnegative")
            if weight > 0:
                cleaned[int(raw_k)] = cleaned.get(int(raw_k), 0.0) + weight
        if not cleaned:
            raise ValueError("at least one block weight must be positive")

        q = constraints.size
        for k in cleaned:
            if q**k > max_patterns:
                raise ValueError(
                    f"pattern cap exceeded for block size {k}: "
                    f"{q}^{k}={q**k} > {max_patterns}"
                )

        block_sizes = np.asarray(sorted(cleaned), dtype=np.int64)
        weights = np.asarray([cleaned[int(k)] for k in block_sizes], dtype=float)
        weights /= weights.sum()

        self.constraints = constraints
        self.block_sizes = block_sizes
        self.block_weights = weights
        self._cumulative_weights = np.cumsum(weights)
        self._outcomes = constraints.outcomes
        self._outcome_to_index = {
            int(value): index for index, value in enumerate(self._outcomes)
        }
        self._groups: dict[int, dict[tuple[int, ...], np.ndarray]] = {}
        self._build_groups()

    def _signature_from_indices(self, pattern: Sequence[int]) -> tuple[int, ...]:
        if self.constraints.features.shape[0] == 0:
            return ()
        totals = self.constraints.features[:, pattern].sum(axis=1)
        return tuple(int(value) for value in totals)

    def _build_groups(self) -> None:
        q = self.constraints.size
        for raw_k in self.block_sizes:
            k = int(raw_k)
            grouped: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
            for pattern in product(range(q), repeat=k):
                signature = self._signature_from_indices(pattern)
                grouped.setdefault(signature, []).append(pattern)
            self._groups[k] = {
                signature: np.asarray(patterns, dtype=np.int16)
                for signature, patterns in grouped.items()
            }

    def contains_pattern(
        self,
        block_size: int,
        signature: Sequence[int],
        outcome_pattern: Sequence[int],
    ) -> bool:
        """Return whether an ordered outcome pattern is in a signature group."""
        try:
            indices = tuple(
                self._outcome_to_index[int(value)] for value in outcome_pattern
            )
        except KeyError:
            return False
        group = self._groups.get(int(block_size), {}).get(tuple(signature))
        if group is None or len(indices) != int(block_size):
            return False
        return bool(np.any(np.all(group == np.asarray(indices), axis=1)))

    def _choose_block_size(self, rng: np.random.Generator) -> int:
        position = int(np.searchsorted(self._cumulative_weights, rng.random(), side="right"))
        return int(self.block_sizes[min(position, len(self.block_sizes) - 1)])

    @staticmethod
    def _sample_source_indices(
        counts: np.ndarray,
        block_size: int,
        rng: np.random.Generator,
    ) -> tuple[int, ...]:
        remaining = counts.copy()
        remaining_total = int(remaining.sum())
        if block_size > remaining_total:
            raise ValueError("block size cannot exceed the number of atoms")
        selected: list[int] = []
        for _ in range(block_size):
            draw = int(rng.integers(remaining_total))
            label_index = int(
                np.searchsorted(np.cumsum(remaining), draw, side="right")
            )
            selected.append(label_index)
            remaining[label_index] -= 1
            remaining_total -= 1
        return tuple(selected)

    def step_inplace(
        self,
        counts: np.ndarray,
        rng: np.random.Generator,
    ) -> StepResult:
        """Apply one exact projected microscopic proposal to ``counts``."""
        block_size = self._choose_block_size(rng)
        source = self._sample_source_indices(counts, block_size, rng)
        signature = self._signature_from_indices(source)
        targets = self._groups[block_size][signature]
        target = tuple(int(value) for value in targets[int(rng.integers(len(targets)))])

        source_hist = np.bincount(source, minlength=self.constraints.size)
        target_hist = np.bincount(target, minlength=self.constraints.size)
        changed = not np.array_equal(source_hist, target_hist)
        counts += target_hist - source_hist

        return StepResult(
            block_size=block_size,
            source_labels=tuple(int(self._outcomes[index]) for index in source),
            target_labels=tuple(int(self._outcomes[index]) for index in target),
            changed=changed,
        )

    def step_atoms_inplace(
        self,
        atoms: np.ndarray,
        counts: np.ndarray,
        rng: np.random.Generator,
    ) -> StepResult:
        """Apply one microscopic proposal to explicit atom positions and counts."""
        block_size = self._choose_block_size(rng)
        positions = rng.choice(len(atoms), size=block_size, replace=False)
        source_values = atoms[positions]
        source = tuple(
            self._outcome_to_index[int(value)] for value in source_values
        )
        signature = self._signature_from_indices(source)
        targets = self._groups[block_size][signature]
        target = tuple(
            int(value) for value in targets[int(rng.integers(len(targets)))]
        )

        source_hist = np.bincount(source, minlength=self.constraints.size)
        target_hist = np.bincount(target, minlength=self.constraints.size)
        changed = not np.array_equal(source_hist, target_hist)
        counts += target_hist - source_hist
        atoms[positions] = self._outcomes[np.asarray(target, dtype=np.int64)]

        return StepResult(
            block_size=block_size,
            source_labels=tuple(int(value) for value in source_values),
            target_labels=tuple(int(self._outcomes[index]) for index in target),
            changed=changed,
        )


def _same_constraints(left: MomentConstraints, right: MomentConstraints) -> bool:
    return (
        np.array_equal(left.outcomes, right.outcomes)
        and np.array_equal(left.features, right.features)
        and np.array_equal(left.targets, right.targets)
    )


def run_chain(
    initial_counts: Sequence[int],
    constraints: MomentConstraints,
    kernel: ReplacementKernel,
    *,
    iterations: int,
    seed: int,
    checkpoint_every: int,
    burn_in: int = 0,
    track_atoms: bool = False,
) -> ChainResult:
    """Run the chain, optionally retaining its real microscopic endpoints."""
    if not _same_constraints(constraints, kernel.constraints):
        raise ValueError("kernel and chain constraints do not match")
    if not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not isinstance(checkpoint_every, (int, np.integer)) or checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be a positive integer")
    if not isinstance(burn_in, (int, np.integer)) or not 0 <= burn_in <= iterations:
        raise ValueError("burn_in must lie between zero and iterations")
    if not isinstance(track_atoms, (bool, np.bool_)):
        raise TypeError("track_atoms must be a boolean")

    counts = constraints.validate_counts(initial_counts).copy()
    if len(kernel.block_sizes) and int(kernel.block_sizes.max()) > int(counts.sum()):
        raise ValueError("a block size exceeds the number of atoms")
    atoms = (
        np.repeat(constraints.outcomes, counts).copy()
        if bool(track_atoms)
        else None
    )
    initial_atoms = atoms.copy() if atoms is not None else None
    rng = np.random.default_rng(seed)
    checkpoint_iterations = [0]
    checkpoint_counts = [counts.copy()]
    proposals_by_block = {int(k): 0 for k in kernel.block_sizes}
    changes_by_block = {int(k): 0 for k in kernel.block_sizes}
    changes = 0

    started = perf_counter()
    for iteration in range(1, int(iterations) + 1):
        step = (
            kernel.step_inplace(counts, rng)
            if atoms is None
            else kernel.step_atoms_inplace(atoms, counts, rng)
        )
        proposals_by_block[step.block_size] += 1
        if step.changed:
            changes += 1
            changes_by_block[step.block_size] += 1
        if iteration % int(checkpoint_every) == 0 or iteration == iterations:
            if atoms is not None:
                atom_counts = np.array(
                    [
                        np.count_nonzero(atoms == value)
                        for value in constraints.outcomes
                    ],
                    dtype=np.int64,
                )
                if not np.array_equal(atom_counts, counts):
                    raise RuntimeError(
                        "tracked atom multiplicities disagree with macro counts"
                    )
            checkpoint_iterations.append(iteration)
            checkpoint_counts.append(counts.copy())
    runtime_seconds = perf_counter() - started

    constraints.validate_counts(counts)
    return ChainResult(
        final_counts=counts.copy(),
        checkpoint_iterations=np.asarray(checkpoint_iterations, dtype=np.int64),
        checkpoint_counts=np.asarray(checkpoint_counts, dtype=np.int64),
        proposals=int(iterations),
        changes=changes,
        burn_in=int(burn_in),
        runtime_seconds=runtime_seconds,
        proposals_by_block=proposals_by_block,
        changes_by_block=changes_by_block,
        initial_atoms=initial_atoms,
        final_atoms=atoms.copy() if atoms is not None else None,
    )


def enumerate_count_transitions(
    counts: Sequence[int],
    kernel: ReplacementKernel,
    block_size: int,
) -> dict[tuple[int, ...], float]:
    """Enumerate one-step count transitions for a fixed block size exactly."""
    if int(block_size) not in kernel._groups:
        raise ValueError("block size is not present in this kernel")
    source_counts = kernel.constraints.validate_counts(counts).copy()
    D = int(source_counts.sum())
    k = int(block_size)
    if k > D:
        raise ValueError("block size cannot exceed the number of atoms")

    q = kernel.constraints.size
    transitions: dict[tuple[int, ...], float] = {}
    for source in product(range(q), repeat=k):
        remaining = source_counts.copy()
        source_probability = 1.0
        remaining_total = D
        for index in source:
            if remaining[index] == 0:
                source_probability = 0.0
                break
            source_probability *= remaining[index] / remaining_total
            remaining[index] -= 1
            remaining_total -= 1
        if source_probability == 0.0:
            continue

        signature = kernel._signature_from_indices(source)
        targets = kernel._groups[k][signature]
        target_probability = source_probability / len(targets)
        source_hist = np.bincount(source, minlength=q)
        for target_array in targets:
            target_hist = np.bincount(target_array, minlength=q)
            new_counts = tuple(
                int(value) for value in source_counts - source_hist + target_hist
            )
            transitions[new_counts] = transitions.get(new_counts, 0.0) + target_probability
    return transitions
