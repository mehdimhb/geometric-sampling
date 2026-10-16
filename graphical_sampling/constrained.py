"""Constrained fixed-size GFS research utilities.

This module keeps hard sample feasibility separate from GFS navigation and
from independent optimization benchmarks.  It intentionally is not exported
from :mod:`graphical_sampling`; prototype callers import it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import fsum
from time import perf_counter
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp

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


@dataclass(frozen=True)
class SearchCounts:
    """Mutually auditable proposal and candidate-state counts."""

    proposed: int
    structurally_valid: int
    no_op: int
    constraint_feasible_candidates: int
    constraint_infeasible_candidates: int
    feasibility_rejected: int
    accepted: int
    non_improving: int
    invariant_failures: int


@dataclass(frozen=True)
class SearchCheckpoint:
    """State of a GFS search after one proposal count."""

    proposal: int
    accepted: int
    objective: float
    infeasible_mass: float
    entropy: float
    support_size: int
    max_fip_error: float
    elapsed_seconds: float


@dataclass(frozen=True)
class GfsSearchResult:
    """Final GFS design, trace, and complete proposal accounting."""

    final_design: Design
    trace: tuple[SearchCheckpoint, ...]
    counts: SearchCounts
    runtime_seconds: float
    status: str
    reached_feasibility: bool


def _validate_search_controls(
    proposals: int, checkpoint_every: int, tolerance: float
) -> tuple[int, int, float]:
    if isinstance(proposals, (bool, np.bool_)) or not isinstance(
        proposals, (int, np.integer)
    ):
        raise TypeError("proposals must be an integer")
    if int(proposals) < 0:
        raise ValueError("proposals must be nonnegative")
    if isinstance(checkpoint_every, (bool, np.bool_)) or not isinstance(
        checkpoint_every, (int, np.integer)
    ):
        raise TypeError("checkpoint_every must be an integer")
    if int(checkpoint_every) <= 0:
        raise ValueError("checkpoint_every must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return int(proposals), int(checkpoint_every), float(tolerance)


def _phase_rng(
    seed: int | np.random.SeedSequence,
) -> np.random.Generator:
    if isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be an integer or SeedSequence")
    if isinstance(seed, (int, np.integer)):
        if int(seed) < 0:
            raise ValueError("seed must be nonnegative")
        return np.random.default_rng(int(seed))
    if not isinstance(seed, np.random.SeedSequence):
        raise TypeError("seed must be an integer or SeedSequence")
    return np.random.default_rng(seed)


def _search_counts(values: Mapping[str, int]) -> SearchCounts:
    return SearchCounts(**{name: int(values[name]) for name in SearchCounts.__dataclass_fields__})


def _record_checkpoint(
    trace: list[SearchCheckpoint],
    *,
    proposal: int,
    accepted: int,
    objective: float,
    diagnostics: DesignDiagnostics,
    started: float,
) -> None:
    row = SearchCheckpoint(
        proposal=int(proposal),
        accepted=int(accepted),
        objective=float(objective),
        infeasible_mass=float(diagnostics.infeasible_mass),
        entropy=float(diagnostics.entropy),
        support_size=int(diagnostics.support_size),
        max_fip_error=float(diagnostics.max_fip_error),
        elapsed_seconds=float(perf_counter() - started),
    )
    if trace and trace[-1].proposal == row.proposal:
        trace[-1] = row
    else:
        trace.append(row)


def run_constraint_repair(
    initial_design: Design,
    prescribed_fip: Sequence[float],
    constraints: ConstraintSystem,
    *,
    proposals: int,
    seed: int | np.random.SeedSequence,
    checkpoint_every: int = 100,
    tolerance: float = 1e-12,
) -> GfsSearchResult:
    """Strictly decrease infeasible mass using maximal inherited GFS exchanges."""

    proposals, checkpoint_every, tolerance = _validate_search_controls(
        proposals, checkpoint_every, tolerance
    )
    if not isinstance(initial_design, Design):
        raise TypeError("initial_design must be a Design")
    rng = _phase_rng(seed)
    started = perf_counter()
    target = np.asarray(prescribed_fip, dtype=np.float64)
    current = initial_design.copy()
    current_diagnostics = validate_probability_mass(
        design_probability_mass(current),
        target,
        constraints,
        require_feasible=False,
        tolerance=tolerance,
    )
    counts = {name: 0 for name in SearchCounts.__dataclass_fields__}
    trace: list[SearchCheckpoint] = []
    _record_checkpoint(
        trace,
        proposal=0,
        accepted=0,
        objective=current_diagnostics.infeasible_mass,
        diagnostics=current_diagnostics,
        started=started,
    )

    reached_feasibility = (
        current_diagnostics.infeasible_mass <= tolerance
        and current_diagnostics.violating_support_size == 0
    )
    if reached_feasibility:
        runtime = perf_counter() - started
        return GfsSearchResult(
            final_design=current,
            trace=tuple(trace),
            counts=_search_counts(counts),
            runtime_seconds=float(runtime),
            status="already_feasible",
            reached_feasibility=True,
        )

    for proposal_index in range(1, proposals + 1):
        counts["proposed"] += 1
        proposal = propose_gfs_exchange(
            current,
            target,
            constraints,
            rng=rng,
            exchange_coef=1.0,
            tolerance=tolerance,
        )
        if proposal.no_op:
            counts["no_op"] += 1
        else:
            counts["structurally_valid"] += 1
            assert proposal.candidate is not None
            assert proposal.diagnostics is not None
            if proposal.constraint_feasible:
                counts["constraint_feasible_candidates"] += 1
            else:
                counts["constraint_infeasible_candidates"] += 1

            if (
                proposal.diagnostics.infeasible_mass
                < current_diagnostics.infeasible_mass - tolerance
            ):
                current = proposal.candidate
                current_diagnostics = proposal.diagnostics
                counts["accepted"] += 1
                _record_checkpoint(
                    trace,
                    proposal=proposal_index,
                    accepted=counts["accepted"],
                    objective=current_diagnostics.infeasible_mass,
                    diagnostics=current_diagnostics,
                    started=started,
                )
            else:
                counts["non_improving"] += 1

        reached_feasibility = (
            current_diagnostics.infeasible_mass <= tolerance
            and current_diagnostics.violating_support_size == 0
        )
        if proposal_index % checkpoint_every == 0 or reached_feasibility:
            _record_checkpoint(
                trace,
                proposal=proposal_index,
                accepted=counts["accepted"],
                objective=current_diagnostics.infeasible_mass,
                diagnostics=current_diagnostics,
                started=started,
            )
        if reached_feasibility:
            break

    _record_checkpoint(
        trace,
        proposal=counts["proposed"],
        accepted=counts["accepted"],
        objective=current_diagnostics.infeasible_mass,
        diagnostics=current_diagnostics,
        started=started,
    )
    runtime = perf_counter() - started
    return GfsSearchResult(
        final_design=current,
        trace=tuple(trace),
        counts=_search_counts(counts),
        runtime_seconds=float(runtime),
        status="feasible" if reached_feasibility else "budget_exhausted",
        reached_feasibility=reached_feasibility,
    )


def enumerate_feasible_samples(
    constraints: ConstraintSystem,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate hard-feasible fixed-size samples in lexicographic order."""

    if not isinstance(constraints, ConstraintSystem):
        raise TypeError("constraints must be a ConstraintSystem")
    return tuple(
        sample
        for sample in combinations(
            range(constraints.population_size), constraints.sample_size
        )
        if is_feasible(sample, constraints)
    )


def incidence_matrix(
    samples: Sequence[Sequence[int]], population_size: int
) -> np.ndarray:
    """Return the population-by-sample binary incidence matrix."""

    size = _validated_population_size(population_size)
    columns = tuple(tuple(sample) for sample in samples)
    result = np.zeros((size, len(columns)), dtype=np.float64)
    for column, sample in enumerate(columns):
        units = tuple(_as_unit(unit, context="sample") for unit in sample)
        if len(set(units)) != len(units):
            raise ValueError("incidence sample units must be distinct")
        if any(unit < 0 or unit >= size for unit in units):
            raise ValueError("incidence sample unit is outside the population range")
        result[np.asarray(units, dtype=np.int64), column] = 1.0
    return result


@dataclass(frozen=True)
class FeasibilityResult:
    """Summary of an independent finite-design LP existence check."""

    success: bool
    status: str
    message: str
    feasible_sample_count: int
    positive_support_size: int
    normalization_error: float
    max_fip_error: float
    max_equality_error: float


def _validated_reference_problem(
    prescribed_fip: Sequence[float],
    samples: Sequence[Sequence[int]],
    incidence: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...], np.ndarray, int]:
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    target, sample_size = _validate_prescribed_fip(
        prescribed_fip, require_positive=False
    )
    normalized_samples = tuple(
        tuple(_as_unit(unit, context="sample") for unit in sample)
        for sample in samples
    )
    if any(len(sample) != sample_size for sample in normalized_samples):
        raise ValueError("every reference sample must have the fixed sample size")
    expected = incidence_matrix(normalized_samples, target.size)
    matrix = np.asarray(incidence, dtype=np.float64)
    if matrix.shape != expected.shape:
        raise ValueError("incidence shape does not match samples and population")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("incidence must contain only finite values")
    if not np.array_equal(matrix, expected):
        raise ValueError("incidence columns are not aligned with samples")
    return target, normalized_samples, matrix, sample_size


def solve_feasibility_reference(
    prescribed_fip: Sequence[float],
    samples: Sequence[Sequence[int]],
    incidence: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> FeasibilityResult:
    """Use HiGHS only to verify that a constrained design exists."""

    target, normalized_samples, matrix, _ = _validated_reference_problem(
        prescribed_fip,
        samples,
        incidence,
        tolerance=tolerance,
    )
    sample_count = len(normalized_samples)
    if sample_count == 0:
        return FeasibilityResult(
            success=False,
            status="infeasible",
            message="hard constraints admit no fixed-size samples",
            feasible_sample_count=0,
            positive_support_size=0,
            normalization_error=float("inf"),
            max_fip_error=float("inf"),
            max_equality_error=float("inf"),
        )

    equality = np.vstack([np.ones(sample_count, dtype=np.float64), matrix])
    rhs = np.concatenate([[1.0], target])
    solution = linprog(
        np.zeros(sample_count, dtype=np.float64),
        A_eq=equality,
        b_eq=rhs,
        bounds=(0.0, None),
        method="highs",
    )
    if not solution.success or solution.x is None:
        if int(solution.status) == 2:
            return FeasibilityResult(
                success=False,
                status="infeasible",
                message=str(solution.message),
                feasible_sample_count=sample_count,
                positive_support_size=0,
                normalization_error=float("inf"),
                max_fip_error=float("inf"),
                max_equality_error=float("inf"),
            )
        raise RuntimeError(
            "HiGHS failed during the feasibility reference: "
            f"status={solution.status}; {solution.message}"
        )

    probability = np.asarray(solution.x, dtype=np.float64)
    if not np.all(np.isfinite(probability)) or np.any(probability < -tolerance):
        raise RuntimeError("HiGHS returned invalid feasibility probabilities")
    normalization_error = abs(float(fsum(float(value) for value in probability)) - 1.0)
    achieved_fip = matrix @ probability
    max_fip_error = float(np.max(np.abs(achieved_fip - target)))
    max_equality_error = float(np.max(np.abs(equality @ probability - rhs)))
    if max(normalization_error, max_fip_error, max_equality_error) > tolerance:
        raise RuntimeError("HiGHS feasibility result failed independent validation")
    return FeasibilityResult(
        success=True,
        status="feasible",
        message=str(solution.message),
        feasible_sample_count=sample_count,
        positive_support_size=int(np.count_nonzero(probability > tolerance)),
        normalization_error=float(normalization_error),
        max_fip_error=max_fip_error,
        max_equality_error=max_equality_error,
    )


@dataclass(frozen=True)
class PairAttainability:
    """Maximum pair SIPs and exhaustive hard/marginal pair taxonomy."""

    maximum_sip: np.ndarray
    categories: Mapping[tuple[int, int], str]
    explicit_forbidden: frozenset[tuple[int, int]]
    other_constraint_incompatible: frozenset[tuple[int, int]]
    marginal_inattainable: frozenset[tuple[int, int]]
    marginal_attainable: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        maximum = np.array(self.maximum_sip, dtype=np.float64, copy=True)
        maximum.flags.writeable = False
        object.__setattr__(self, "maximum_sip", maximum)
        object.__setattr__(self, "categories", MappingProxyType(dict(self.categories)))
        for name in (
            "explicit_forbidden",
            "other_constraint_incompatible",
            "marginal_inattainable",
            "marginal_attainable",
        ):
            object.__setattr__(self, name, frozenset(getattr(self, name)))


def analyze_pair_attainability(
    prescribed_fip: Sequence[float],
    samples: Sequence[Sequence[int]],
    incidence: np.ndarray,
    constraints: ConstraintSystem,
    *,
    tolerance: float = 1e-12,
) -> PairAttainability:
    """Maximize each compatible pair SIP under the prescribed FIP margins."""

    if not isinstance(constraints, ConstraintSystem):
        raise TypeError("constraints must be a ConstraintSystem")
    target, normalized_samples, matrix, _ = _validated_reference_problem(
        prescribed_fip,
        samples,
        incidence,
        tolerance=tolerance,
    )
    if (
        constraints.population_size != target.size
        or normalized_samples != enumerate_feasible_samples(constraints)
    ):
        raise ValueError(
            "samples must be the complete enumerated hard-feasible sample space"
        )
    feasibility = solve_feasibility_reference(
        target,
        normalized_samples,
        matrix,
        tolerance=tolerance,
    )
    if not feasibility.success:
        raise ValueError(
            "pair attainability is undefined because the prescribed constrained "
            "design is infeasible"
        )

    sample_count = len(normalized_samples)
    equality = np.vstack([np.ones(sample_count, dtype=np.float64), matrix])
    rhs = np.concatenate([[1.0], target])
    maximum = np.zeros((target.size, target.size), dtype=np.float64)
    np.fill_diagonal(maximum, target)
    categories: dict[tuple[int, int], str] = {}
    category_sets: dict[str, set[tuple[int, int]]] = {
        "explicit_forbidden": set(),
        "other_constraint_incompatible": set(),
        "marginal_inattainable": set(),
        "marginal_attainable": set(),
    }
    explicit_pairs = {
        tuple(sorted(int(unit) for unit in pair))
        for pair in constraints.forbidden_pairs
    }

    for first, second in combinations(range(target.size), 2):
        pair = (first, second)
        if pair in explicit_pairs:
            category = "explicit_forbidden"
            optimum = 0.0
        else:
            pair_incidence = matrix[first] * matrix[second]
            if not np.any(pair_incidence):
                category = "other_constraint_incompatible"
                optimum = 0.0
            else:
                solution = linprog(
                    -pair_incidence,
                    A_eq=equality,
                    b_eq=rhs,
                    bounds=(0.0, None),
                    method="highs",
                )
                if not solution.success or solution.x is None:
                    raise RuntimeError(
                        "HiGHS failed during pair-attainability analysis for "
                        f"{pair}: status={solution.status}; {solution.message}"
                    )
                probability = np.asarray(solution.x, dtype=np.float64)
                residual = float(np.max(np.abs(equality @ probability - rhs)))
                if (
                    not np.all(np.isfinite(probability))
                    or np.any(probability < -tolerance)
                    or residual > tolerance
                ):
                    raise RuntimeError(
                        f"pair-attainability solution for {pair} failed validation"
                    )
                optimum = float(pair_incidence @ probability)
                if (
                    not np.isfinite(optimum)
                    or optimum < -tolerance
                    or optimum > 1.0 + tolerance
                    or abs(optimum + float(solution.fun)) > tolerance
                ):
                    raise RuntimeError(
                        f"pair-attainability objective for {pair} failed validation"
                    )
                category = (
                    "marginal_attainable"
                    if optimum > tolerance
                    else "marginal_inattainable"
                )
        maximum[first, second] = optimum
        maximum[second, first] = optimum
        categories[pair] = category
        category_sets[category].add(pair)

    return PairAttainability(
        maximum_sip=maximum,
        categories=categories,
        explicit_forbidden=frozenset(category_sets["explicit_forbidden"]),
        other_constraint_incompatible=frozenset(
            category_sets["other_constraint_incompatible"]
        ),
        marginal_inattainable=frozenset(
            category_sets["marginal_inattainable"]
        ),
        marginal_attainable=frozenset(category_sets["marginal_attainable"]),
    )


@dataclass(frozen=True)
class MaxEntReference:
    """Validated numerical maximum-entropy reference over feasible samples."""

    probabilities: np.ndarray
    entropy: float
    max_equality_error: float
    success: bool
    message: str
    label: str = "Numerical constrained MaxEnt benchmark"

    def __post_init__(self) -> None:
        probability = np.array(self.probabilities, dtype=np.float64, copy=True)
        probability.flags.writeable = False
        object.__setattr__(self, "probabilities", probability)


def solve_numerical_maxent(
    prescribed_fip: Sequence[float],
    samples: Sequence[Sequence[int]],
    incidence: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> MaxEntReference:
    """Solve and independently validate the constrained finite MaxEnt dual."""

    target, normalized_samples, matrix, _ = _validated_reference_problem(
        prescribed_fip,
        samples,
        incidence,
        tolerance=tolerance,
    )
    feasibility = solve_feasibility_reference(
        target,
        normalized_samples,
        matrix,
        tolerance=tolerance,
    )
    if not feasibility.success:
        raise ValueError(
            "Numerical constrained MaxEnt benchmark is mathematically infeasible"
        )

    features = matrix[:-1]
    reduced_target = target[:-1]

    def dual_terms(
        theta: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        logits = theta @ features
        log_partition = float(logsumexp(logits))
        probability = np.exp(logits - log_partition)
        moments = features @ probability
        gradient = moments - reduced_target
        hessian = (features * probability) @ features.T - np.outer(
            moments, moments
        )
        objective = log_partition - float(theta @ reduced_target)
        return objective, gradient, hessian, probability

    def objective(theta: np.ndarray) -> float:
        return dual_terms(theta)[0]

    def gradient(theta: np.ndarray) -> np.ndarray:
        return dual_terms(theta)[1]

    def hessian(theta: np.ndarray) -> np.ndarray:
        return dual_terms(theta)[2]

    solution = minimize(
        objective,
        np.zeros(features.shape[0], dtype=np.float64),
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": min(tolerance / 10.0, 1e-12), "maxiter": 10_000},
    )
    _, _, _, probability = dual_terms(np.asarray(solution.x, dtype=np.float64))
    equality = np.vstack(
        [np.ones(len(normalized_samples), dtype=np.float64), matrix]
    )
    rhs = np.concatenate([[1.0], target])
    residual = equality @ probability - rhs
    max_equality_error = float(np.max(np.abs(residual)))
    if (
        not np.all(np.isfinite(solution.x))
        or not np.all(np.isfinite(probability))
        or np.any(probability < 0.0)
        or max_equality_error > tolerance
    ):
        raise RuntimeError(
            "Numerical constrained MaxEnt benchmark failed to reach tolerance: "
            f"{solution.message}; residual={max_equality_error:.3e}"
        )
    entropy = float(
        -np.sum(probability[probability > 0.0] * np.log(probability[probability > 0.0]))
    )
    if not np.isfinite(entropy):
        raise RuntimeError("Numerical constrained MaxEnt benchmark returned nonfinite entropy")
    return MaxEntReference(
        probabilities=probability,
        entropy=entropy,
        max_equality_error=max_equality_error,
        success=True,
        message=str(solution.message),
    )


def attainable_pair_score(
    mass: Mapping[tuple[int, ...], float],
    pairs: Iterable[tuple[int, int]],
    *,
    offset: float = 1e-12,
) -> float:
    """Score only the supplied marginal-attainable unordered pairs."""

    if not np.isfinite(offset) or offset <= 0.0:
        raise ValueError("offset must be finite and positive")
    if not isinstance(mass, Mapping):
        raise TypeError("mass must be a mapping from samples to probabilities")
    normalized_pairs: list[tuple[int, int]] = []
    for raw_pair in pairs:
        try:
            units = tuple(_as_unit(unit, context="pairs") for unit in raw_pair)
        except TypeError:
            raise
        except Exception as exc:
            raise TypeError("pairs must contain two-element iterables") from exc
        if len(units) != 2 or units[0] == units[1]:
            raise ValueError("each attainable pair must contain two distinct units")
        pair = tuple(sorted(units))
        if pair[0] < 0:
            raise ValueError("attainable pair indices must be nonnegative")
        normalized_pairs.append(pair)
    if len(set(normalized_pairs)) != len(normalized_pairs):
        raise ValueError("attainable pairs must not contain duplicates")

    support: list[tuple[frozenset[int], float]] = []
    for raw_sample, raw_probability in mass.items():
        raw_units = tuple(raw_sample)
        sample = frozenset(
            _as_unit(unit, context="support sample") for unit in raw_units
        )
        if len(sample) != len(raw_units):
            raise ValueError("support sample units must be distinct")
        probability = float(raw_probability)
        if not np.isfinite(probability) or probability < 0.0:
            raise ValueError("sample probabilities must be finite and nonnegative")
        support.append((sample, probability))

    terms = []
    for pair in normalized_pairs:
        pair_set = frozenset(pair)
        sip = fsum(
            probability
            for sample, probability in support
            if pair_set <= sample
        )
        terms.append(float(np.log(float(sip) + float(offset))))
    score = float(fsum(terms))
    if not np.isfinite(score):
        raise ValueError("attainable-pair score is nonfinite")
    return score


def run_feasible_gfs_search(
    initial_design: Design,
    prescribed_fip: Sequence[float],
    constraints: ConstraintSystem,
    *,
    objective: Literal["entropy", "sip"],
    attainable_pairs: Iterable[tuple[int, int]] = (),
    proposals: int,
    seed: int | np.random.SeedSequence,
    checkpoint_every: int = 100,
    tolerance: float = 1e-12,
) -> GfsSearchResult:
    """Hill-climb a hard-feasible design using inherited GFS exchanges only."""

    proposals, checkpoint_every, tolerance = _validate_search_controls(
        proposals, checkpoint_every, tolerance
    )
    if not isinstance(initial_design, Design):
        raise TypeError("initial_design must be a Design")
    target = np.asarray(prescribed_fip, dtype=np.float64)
    current = initial_design.copy()
    try:
        current_diagnostics = validate_probability_mass(
            design_probability_mass(current),
            target,
            constraints,
            require_feasible=True,
            tolerance=tolerance,
        )
    except ValueError as exc:
        raise ValueError(
            "Stage 2 requires a feasible initial design with valid invariants"
        ) from exc
    if objective not in ("entropy", "sip"):
        raise ValueError("objective must be either 'entropy' or 'sip'")
    normalized_pairs = tuple(attainable_pairs)

    rng = _phase_rng(seed)
    started = perf_counter()
    current_objective = (
        current_diagnostics.entropy
        if objective == "entropy"
        else attainable_pair_score(
            design_probability_mass(current), normalized_pairs
        )
    )
    counts = {name: 0 for name in SearchCounts.__dataclass_fields__}
    trace: list[SearchCheckpoint] = []
    _record_checkpoint(
        trace,
        proposal=0,
        accepted=0,
        objective=current_objective,
        diagnostics=current_diagnostics,
        started=started,
    )

    for proposal_index in range(1, proposals + 1):
        counts["proposed"] += 1
        alpha = float(rng.uniform(0.1, 0.9))
        proposal = propose_gfs_exchange(
            current,
            target,
            constraints,
            rng=rng,
            exchange_coef=alpha,
            tolerance=tolerance,
        )
        if proposal.no_op:
            counts["no_op"] += 1
        else:
            counts["structurally_valid"] += 1
            assert proposal.candidate is not None
            assert proposal.diagnostics is not None
            if not proposal.constraint_feasible:
                counts["constraint_infeasible_candidates"] += 1
                counts["feasibility_rejected"] += 1
            else:
                counts["constraint_feasible_candidates"] += 1
                candidate_objective = (
                    proposal.diagnostics.entropy
                    if objective == "entropy"
                    else attainable_pair_score(
                        design_probability_mass(proposal.candidate),
                        normalized_pairs,
                    )
                )
                if not np.isfinite(candidate_objective):
                    raise RuntimeError("candidate objective is nonfinite")
                if candidate_objective > current_objective + tolerance:
                    current = proposal.candidate
                    current_diagnostics = proposal.diagnostics
                    current_objective = float(candidate_objective)
                    counts["accepted"] += 1
                    _record_checkpoint(
                        trace,
                        proposal=proposal_index,
                        accepted=counts["accepted"],
                        objective=current_objective,
                        diagnostics=current_diagnostics,
                        started=started,
                    )
                else:
                    counts["non_improving"] += 1

        if proposal_index % checkpoint_every == 0:
            _record_checkpoint(
                trace,
                proposal=proposal_index,
                accepted=counts["accepted"],
                objective=current_objective,
                diagnostics=current_diagnostics,
                started=started,
            )

    _record_checkpoint(
        trace,
        proposal=counts["proposed"],
        accepted=counts["accepted"],
        objective=current_objective,
        diagnostics=current_diagnostics,
        started=started,
    )
    runtime = perf_counter() - started
    return GfsSearchResult(
        final_design=current,
        trace=tuple(trace),
        counts=_search_counts(counts),
        runtime_seconds=float(runtime),
        status="completed",
        reached_feasibility=True,
    )
