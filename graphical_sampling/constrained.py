"""Constrained fixed-size GFS research utilities.

This module keeps hard sample feasibility separate from GFS navigation and
from independent optimization benchmarks.  It intentionally is not exported
from :mod:`graphical_sampling`; prototype callers import it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import fsum
from typing import Iterable, Mapping, Sequence

import numpy as np

from .design import Design
from .maxent.finite_population import design_probability_mass
from .order import Cluster, Order, Zone
from .population import Population
from .structs import _Sample


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


def role_rng(root_seed: int, role: int) -> np.random.Generator:
    """Create an isolated deterministic RNG stream for one experiment role."""

    for name, value in (("root_seed", root_seed), ("role", role)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
        if int(value) < 0:
            raise ValueError(f"{name} must be nonnegative")
    return np.random.default_rng(
        np.random.SeedSequence([int(root_seed), int(role)])
    )


def seeded_madow_order(
    population_size: int, root_seed: int = 12345
) -> np.ndarray:
    """Generate the role-0 Madow order without constraint or objective input."""

    size = _validated_population_size(population_size)
    return role_rng(root_seed, 0).permutation(size)


def _validate_prescribed_fip(
    prescribed_fip: Sequence[float], *, require_positive: bool
) -> tuple[np.ndarray, int]:
    try:
        target = np.asarray(prescribed_fip, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("prescribed_fip must be a numeric one-dimensional vector") from exc
    if target.ndim != 1 or target.size == 0:
        raise ValueError("prescribed_fip must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(target)):
        raise ValueError("prescribed_fip must contain only finite values")
    lower_invalid = target <= 0.0 if require_positive else target < 0.0
    if np.any(lower_invalid) or np.any(target > 1.0):
        interval = "(0, 1]" if require_positive else "[0, 1]"
        raise ValueError(f"prescribed_fip values must lie in {interval}")
    total = float(fsum(float(value) for value in target))
    sample_size = int(round(total))
    if (
        sample_size <= 0
        or sample_size > target.size
        or not np.isclose(total, sample_size, atol=1e-12, rtol=0.0)
    ):
        raise ValueError("prescribed_fip must sum to an integer sample size")
    return np.array(target, dtype=np.float64, copy=True), sample_size


class _Float64Design(Design):
    """Minimal precision adapter retaining the inherited GFS transition."""

    def copy(self) -> _Float64Design:
        clone = super().copy()
        clone.__class__ = type(self)
        return clone

    def _push(self, part_idx: int, *args: _Sample) -> None:
        for sample in args:
            probability = float(sample.prob)
            if not np.isfinite(probability):
                raise ValueError("GFS fragment probability must be finite")
            if probability < 0.0:
                raise ValueError("GFS fragment probability must be nonnegative")
            if probability > 0.0:
                self._heaps[part_idx].push(sample)


def build_madow_design(
    prescribed_fip: Sequence[float],
    coordinates: np.ndarray,
    order: Sequence[int],
) -> Design:
    """Build the package Madow/GFS design in a predeclared population order."""

    target, sample_size = _validate_prescribed_fip(
        prescribed_fip, require_positive=True
    )
    population_size = target.size

    try:
        coords = np.asarray(coordinates, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinates must be a numeric array with shape (N, 2)") from exc
    if coords.shape != (population_size, 2):
        raise ValueError("coordinates must have shape (N, 2)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates must contain only finite values")

    try:
        raw_order = tuple(order)
    except TypeError as exc:
        raise TypeError("order must be an iterable of population indices") from exc
    permutation = np.asarray(
        [_as_unit(unit, context="order") for unit in raw_order], dtype=np.int64
    )
    if permutation.shape != (population_size,):
        raise ValueError("order must contain every population index exactly once")
    if set(permutation.tolist()) != set(range(population_size)):
        raise ValueError("order must be a permutation of the population indices")

    population = Population(
        coords,
        target,
        variable=np.ones(population_size, dtype=np.float64),
        n=None,
    )
    population._n = sample_size
    zone = Zone(
        _shares=np.ones(population_size, dtype=np.float64),
        _indices=permutation,
        sort=list(range(population_size)),
    )
    madow_order = Order(population)
    madow_order.clusters = [Cluster(label=0, zones=[zone])]
    madow_order.num_zones = 1
    madow_order.num_splits = 1
    madow_order._build_order(1)

    design = _Float64Design.from_order(population, madow_order)
    validate_probability_mass(
        design_probability_mass(design),
        target,
        ConstraintSystem(population_size, sample_size),
        tolerance=1e-12,
    )
    return design


@dataclass(frozen=True)
class ExchangeProposal:
    """Auditable result of one inherited package GFS exchange proposal."""

    candidate: Design | None
    diagnostics: DesignDiagnostics | None
    no_op: bool
    constraint_feasible: bool


def _mass_maps_close(
    first: Mapping[tuple[int, ...], float],
    second: Mapping[tuple[int, ...], float],
    *,
    tolerance: float,
) -> bool:
    keys = set(first) | set(second)
    return all(abs(first.get(key, 0.0) - second.get(key, 0.0)) <= tolerance for key in keys)


def propose_gfs_exchange(
    design: Design,
    prescribed_fip: Sequence[float],
    constraints: ConstraintSystem,
    *,
    rng: np.random.Generator,
    exchange_coef: float,
    tolerance: float = 1e-12,
) -> ExchangeProposal:
    """Propose one inherited Algorithm-4 exchange on a deterministic copy."""

    if not isinstance(design, Design):
        raise TypeError("design must be a Design")
    if not isinstance(constraints, ConstraintSystem):
        raise TypeError("constraints must be a ConstraintSystem")
    if design.pop.N != constraints.population_size or design.pop.n != constraints.sample_size:
        raise ValueError("design dimensions must match the constraint system")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not np.isfinite(exchange_coef) or not 0.0 < exchange_coef <= 1.0:
        raise ValueError("exchange_coef must lie in (0, 1]")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    target = np.asarray(prescribed_fip, dtype=np.float64)
    before = design_probability_mass(design)
    validate_probability_mass(
        before,
        target,
        constraints,
        require_feasible=False,
        tolerance=tolerance,
    )

    candidate = design.copy()
    seeds = rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=1 + len(candidate._heaps),
        dtype=np.int64,
    )
    candidate._rng = np.random.default_rng(int(seeds[0]))
    for heap, seed in zip(candidate._heaps, seeds[1:]):
        heap.rng = np.random.default_rng(int(seed))

    candidate.exchange(
        partitions=1,
        pull_strategy="random",
        exchange_coef=float(exchange_coef),
    )
    after = design_probability_mass(candidate)
    if _mass_maps_close(before, after, tolerance=tolerance):
        return ExchangeProposal(
            candidate=None,
            diagnostics=None,
            no_op=True,
            constraint_feasible=False,
        )

    diagnostics = validate_probability_mass(
        after,
        target,
        constraints,
        require_feasible=False,
        tolerance=tolerance,
    )
    return ExchangeProposal(
        candidate=candidate,
        diagnostics=diagnostics,
        no_op=False,
        constraint_feasible=diagnostics.violating_support_size == 0,
    )
