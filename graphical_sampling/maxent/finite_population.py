"""Unconditional graphical exchanges for fixed-size sampling designs."""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Sequence

import numpy as np

from ..design import Design


@dataclass(frozen=True)
class FiniteDesignDiagnostic:
    """Read-only summary of a finite design after a number of exchanges."""

    exchange: int
    entropy: float
    effective_support: float
    support_size: int
    probability_sum_error: float
    fip_error: float


@dataclass(frozen=True)
class FiniteExchangeResult:
    """Final exchanged design and its checkpoint diagnostics."""

    final_design: Design
    trace: tuple[FiniteDesignDiagnostic, ...]


def design_probability_mass(design: Design) -> dict[tuple[int, ...], float]:
    """Aggregate equal samples without mutating or merging ``design``."""
    pieces: dict[tuple[int, ...], list[float]] = {}
    for heap in design._heaps:
        for sample in heap:
            key = tuple(sorted(int(unit) for unit in sample.ids))
            pieces.setdefault(key, []).append(float(sample.prob))
    return {sample: fsum(values) for sample, values in pieces.items()}


def _diagnose_design(
    design: Design,
    prescribed_fip: np.ndarray,
    *,
    exchange: int,
    tolerance: float,
) -> FiniteDesignDiagnostic:
    mass = design_probability_mass(design)
    if not mass:
        raise ValueError("design support must not be empty")
    if any(len(sample) != design.pop.n for sample in mass):
        raise ValueError("every support sample must have the fixed sample size")

    probability = np.asarray(list(mass.values()), dtype=float)
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("sample probabilities must be finite and nonnegative")
    probability_sum_error = abs(fsum(float(value) for value in probability) - 1.0)
    if probability_sum_error > tolerance:
        raise ValueError("sample probabilities do not sum to one within tolerance")

    fip = np.zeros(design.pop.N, dtype=float)
    for sample, sample_probability in mass.items():
        indices = np.asarray(sample, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= design.pop.N):
            raise ValueError("support sample contains an invalid population index")
        fip[indices] += sample_probability
    fip_error = float(np.max(np.abs(fip - prescribed_fip)))
    if fip_error > tolerance:
        raise ValueError("design does not preserve prescribed FIPs within tolerance")

    positive = probability > 0.0
    entropy = float(-np.sum(probability[positive] * np.log(probability[positive])))
    return FiniteDesignDiagnostic(
        exchange=int(exchange),
        entropy=entropy,
        effective_support=float(np.exp(entropy)),
        support_size=int(np.count_nonzero(probability > tolerance)),
        probability_sum_error=float(probability_sum_error),
        fip_error=fip_error,
    )


def run_design_exchanges(
    initial_design: Design,
    prescribed_fip: Sequence[float],
    *,
    exchanges: int,
    seed: int,
    checkpoint_every: int,
    partitions: int = 1,
    pull_strategy: str = "default",
    exchange_coef: float = 0.75,
    tolerance: float = 1e-12,
) -> FiniteExchangeResult:
    """Apply every requested package exchange and record read-only diagnostics."""
    if not isinstance(initial_design, Design):
        raise TypeError("initial_design must be a Design")
    if not isinstance(exchanges, (int, np.integer)) or isinstance(exchanges, bool):
        raise ValueError("exchanges must be a nonnegative integer")
    if exchanges < 0:
        raise ValueError("exchanges must be a nonnegative integer")
    if (
        not isinstance(checkpoint_every, (int, np.integer))
        or isinstance(checkpoint_every, bool)
        or checkpoint_every <= 0
    ):
        raise ValueError("checkpoint_every must be a positive integer")
    if (
        not isinstance(partitions, (int, np.integer))
        or isinstance(partitions, bool)
        or partitions <= 0
    ):
        raise ValueError("partitions must be a positive integer")
    if not np.isfinite(exchange_coef) or not 0.0 < exchange_coef <= 1.0:
        raise ValueError("exchange_coef must lie in (0, 1]")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    target = np.asarray(prescribed_fip, dtype=float)
    if target.shape != (initial_design.pop.N,):
        raise ValueError("prescribed_fip length must equal the population size")
    if not np.all(np.isfinite(target)):
        raise ValueError("prescribed_fip must contain only finite values")

    final_design = initial_design.copy()
    children = np.random.SeedSequence(seed).spawn(1 + len(final_design._heaps))
    final_design._rng = np.random.default_rng(children[0])
    for heap, child in zip(final_design._heaps, children[1:]):
        heap.rng = np.random.default_rng(child)

    trace = [
        _diagnose_design(
            final_design,
            target,
            exchange=0,
            tolerance=tolerance,
        )
    ]
    for exchange in range(1, int(exchanges) + 1):
        final_design.exchange(
            partitions=int(partitions),
            pull_strategy=pull_strategy,
            exchange_coef=float(exchange_coef),
        )
        if exchange % int(checkpoint_every) == 0 or exchange == int(exchanges):
            trace.append(
                _diagnose_design(
                    final_design,
                    target,
                    exchange=exchange,
                    tolerance=tolerance,
                )
            )

    return FiniteExchangeResult(final_design=final_design, trace=tuple(trace))
