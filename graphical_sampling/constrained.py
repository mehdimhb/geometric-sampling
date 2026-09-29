"""Constrained fixed-size GFS research utilities.

This module keeps hard sample feasibility separate from GFS navigation and
from independent optimization benchmarks.  It intentionally is not exported
from :mod:`graphical_sampling`; prototype callers import it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import fsum
from typing import Iterable, Mapping

import numpy as np


def _as_unit(value: object, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{context} must contain integer population indices")
    return int(value)


@dataclass(frozen=True)
class GroupCapacity:
    """Maximum number of units that a sample may take from one group."""

    units: frozenset[int]
    capacity: int

    def __post_init__(self) -> None:
        try:
            units = frozenset(
                _as_unit(unit, context="group units") for unit in self.units
            )
        except TypeError:
            raise
        except Exception as exc:
            raise TypeError("group units must be an iterable of indices") from exc
        if not units:
            raise ValueError("group units must be nonempty")
        if isinstance(self.capacity, (bool, np.bool_)) or not isinstance(
            self.capacity, (int, np.integer)
        ):
            raise TypeError("group capacity must be an integer")
        capacity = int(self.capacity)
        if capacity < 0:
            raise ValueError("group capacity must be nonnegative")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "capacity", capacity)


@dataclass(frozen=True)
class ConstraintSystem:
    """Fixed-size sample constraints over zero-based population indices."""

    population_size: int
    sample_size: int
    forbidden_pairs: frozenset[frozenset[int]] = frozenset()
    group_capacities: tuple[GroupCapacity, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("population_size", self.population_size),
            ("sample_size", self.sample_size),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
        population_size = int(self.population_size)
        sample_size = int(self.sample_size)
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if sample_size <= 0 or sample_size > population_size:
            raise ValueError("sample_size must lie between one and population_size")

        try:
            pairs = []
            for raw_pair in self.forbidden_pairs:
                pair = frozenset(
                    _as_unit(unit, context="forbidden pairs") for unit in raw_pair
                )
                if len(pair) != 2:
                    raise ValueError(
                        "each forbidden pair must contain two distinct indices"
                    )
                pairs.append(pair)
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise TypeError("forbidden_pairs must be an iterable of pairs") from exc
        forbidden_pairs = frozenset(pairs)

        try:
            capacities = tuple(self.group_capacities)
        except Exception as exc:
            raise TypeError("group_capacities must be an iterable") from exc
        if any(not isinstance(item, GroupCapacity) for item in capacities):
            raise TypeError("every group capacity must be a GroupCapacity")

        valid_units = range(population_size)
        if any(unit not in valid_units for pair in forbidden_pairs for unit in pair):
            raise ValueError("forbidden-pair unit is outside the population range")
        if any(
            unit not in valid_units
            for capacity in capacities
            for unit in capacity.units
        ):
            raise ValueError("group-capacity unit is outside the population range")

        object.__setattr__(self, "population_size", population_size)
        object.__setattr__(self, "sample_size", sample_size)
        object.__setattr__(self, "forbidden_pairs", forbidden_pairs)
        object.__setattr__(self, "group_capacities", capacities)


def _validated_sample(
    sample: Iterable[int], constraints: ConstraintSystem
) -> frozenset[int]:
    if not isinstance(constraints, ConstraintSystem):
        raise TypeError("constraints must be a ConstraintSystem")
    try:
        ordered = tuple(_as_unit(unit, context="sample") for unit in sample)
    except TypeError:
        raise
    except Exception as exc:
        raise TypeError("sample must be an iterable of population indices") from exc
    units = frozenset(ordered)
    if len(units) != len(ordered):
        raise ValueError("sample units must be distinct")
    if len(units) != constraints.sample_size:
        raise ValueError("sample does not have the required fixed sample size")
    if any(unit < 0 or unit >= constraints.population_size for unit in units):
        raise ValueError("sample unit is outside the population range")
    return units


def is_feasible(sample: Iterable[int], constraints: ConstraintSystem) -> bool:
    """Return whether a well-formed fixed-size sample satisfies all constraints."""

    units = _validated_sample(sample, constraints)
    if any(pair <= units for pair in constraints.forbidden_pairs):
        return False
    return all(
        len(units & capacity.units) <= capacity.capacity
        for capacity in constraints.group_capacities
    )


def _validated_population_size(population_size: int) -> int:
    if isinstance(population_size, (bool, np.bool_)) or not isinstance(
        population_size, (int, np.integer)
    ):
        raise TypeError("population_size must be an integer")
    result = int(population_size)
    if result <= 0:
        raise ValueError("population_size must be positive")
    return result


def _mass_items(
    mass: Mapping[tuple[int, ...], float], population_size: int
) -> tuple[tuple[tuple[int, ...], float], ...]:
    if not isinstance(mass, Mapping):
        raise TypeError("mass must be a mapping from samples to probabilities")
    items: list[tuple[tuple[int, ...], float]] = []
    for raw_sample, raw_probability in mass.items():
        try:
            sample = tuple(_as_unit(unit, context="support sample") for unit in raw_sample)
        except TypeError:
            raise
        except Exception as exc:
            raise TypeError("support samples must be iterable") from exc
        if len(set(sample)) != len(sample):
            raise ValueError("support sample units must be distinct")
        if any(unit < 0 or unit >= population_size for unit in sample):
            raise ValueError("support sample unit is outside the population range")
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError) as exc:
            raise TypeError("sample probabilities must be real numbers") from exc
        if not np.isfinite(probability):
            raise ValueError("sample probabilities must be finite")
        if probability < 0.0:
            raise ValueError("sample probabilities must be nonnegative")
        items.append((sample, probability))
    return tuple(items)


def compute_fip(
    mass: Mapping[tuple[int, ...], float], population_size: int
) -> np.ndarray:
    """Compute first-order inclusion probabilities from finite support mass."""

    size = _validated_population_size(population_size)
    result = np.zeros(size, dtype=np.float64)
    for sample, probability in _mass_items(mass, size):
        result[np.asarray(sample, dtype=np.int64)] += probability
    return result


def compute_sip(
    mass: Mapping[tuple[int, ...], float], population_size: int
) -> np.ndarray:
    """Compute the exact symmetric second-order inclusion matrix."""

    size = _validated_population_size(population_size)
    result = np.zeros((size, size), dtype=np.float64)
    for sample, probability in _mass_items(mass, size):
        for unit in sample:
            result[unit, unit] += probability
        for first, second in combinations(sample, 2):
            result[first, second] += probability
            result[second, first] += probability
    return result


def design_entropy(mass: Mapping[tuple[int, ...], float]) -> float:
    """Return Shannon entropy of the positive finite support probabilities."""

    if not isinstance(mass, Mapping):
        raise TypeError("mass must be a mapping from samples to probabilities")
    probabilities = np.asarray(tuple(mass.values()), dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("sample probabilities must be finite")
    if np.any(probabilities < 0.0):
        raise ValueError("sample probabilities must be nonnegative")
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def infeasible_mass(
    mass: Mapping[tuple[int, ...], float], constraints: ConstraintSystem
) -> float:
    """Return total probability assigned to hard-constraint-infeasible samples."""

    items = _mass_items(mass, constraints.population_size)
    return float(
        fsum(
            probability
            for sample, probability in items
            if not is_feasible(sample, constraints)
        )
    )


@dataclass(frozen=True)
class DesignDiagnostics:
    """Validated numerical summary of one finite fixed-size design."""

    probability_sum_error: float
    max_fip_error: float
    support_size: int
    violating_support_size: int
    infeasible_mass: float
    entropy: float
    fip: np.ndarray
    sip: np.ndarray

    def __post_init__(self) -> None:
        for name in ("fip", "sip"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            values.flags.writeable = False
            object.__setattr__(self, name, values)


def validate_probability_mass(
    mass: Mapping[tuple[int, ...], float],
    prescribed_fip: Iterable[float],
    constraints: ConstraintSystem,
    *,
    require_feasible: bool = False,
    tolerance: float = 1e-12,
) -> DesignDiagnostics:
    """Validate core invariants and return diagnostics without repairing mass."""

    if not isinstance(constraints, ConstraintSystem):
        raise TypeError("constraints must be a ConstraintSystem")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(require_feasible, (bool, np.bool_)):
        raise TypeError("require_feasible must be boolean")

    try:
        target = np.asarray(prescribed_fip, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("prescribed_fip must be a numeric vector") from exc
    if target.shape != (constraints.population_size,):
        raise ValueError("prescribed_fip must match the population size")
    if not np.all(np.isfinite(target)):
        raise ValueError("prescribed_fip must contain only finite values")
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("prescribed_fip values must lie in [0, 1]")

    items = _mass_items(mass, constraints.population_size)
    if not items:
        raise ValueError("design support must not be empty")
    if any(len(sample) != constraints.sample_size for sample, _ in items):
        raise ValueError("every support sample must have the fixed sample size")
    probability_sum_error = abs(
        fsum(probability for _, probability in items) - 1.0
    )
    if probability_sum_error > tolerance:
        raise ValueError("sample probabilities do not sum to one within tolerance")

    canonical_mass = {sample: probability for sample, probability in items}
    fip = compute_fip(canonical_mass, constraints.population_size)
    max_fip_error = float(np.max(np.abs(fip - target)))
    if max_fip_error > tolerance:
        raise ValueError("design does not preserve the prescribed FIPs")

    violating = tuple(
        (sample, probability)
        for sample, probability in items
        if probability > tolerance and not is_feasible(sample, constraints)
    )
    violation_mass = float(fsum(probability for _, probability in violating))
    if require_feasible and violating:
        raise ValueError("design support violates the hard constraints")

    return DesignDiagnostics(
        probability_sum_error=float(probability_sum_error),
        max_fip_error=max_fip_error,
        support_size=sum(probability > tolerance for _, probability in items),
        violating_support_size=len(violating),
        infeasible_mass=violation_mass,
        entropy=design_entropy(canonical_mass),
        fip=fip,
        sip=compute_sip(canonical_mass, constraints.population_size),
    )
