"""Validated information-theoretic diagnostics for finite distributions."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .constraints import MomentConstraints


def _validate_probability(
    values: Sequence[float],
    *,
    name: str = "probability",
) -> np.ndarray:
    probability = np.asarray(values, dtype=float)
    if probability.ndim != 1 or len(probability) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{name} must be finite")
    if np.any(probability < 0):
        raise ValueError(f"{name} must be nonnegative")
    if not np.isclose(probability.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return probability


def shannon_entropy(probability: Sequence[float]) -> float:
    """Return Shannon entropy in nats, with zero-mass terms omitted."""
    p = _validate_probability(probability)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log(p[positive])))


def kl_divergence(
    probability: Sequence[float],
    benchmark: Sequence[float],
) -> float:
    """Return ``KL(probability || benchmark)`` in nats."""
    p = _validate_probability(probability)
    q = _validate_probability(benchmark, name="benchmark")
    if p.shape != q.shape:
        raise ValueError("probability and benchmark must have the same shape")
    positive = p > 0
    if np.any(q[positive] <= 0):
        raise ValueError("benchmark must be positive wherever probability is positive")
    return float(np.sum(p[positive] * np.log(p[positive] / q[positive])))


def _counts_or_probability(
    values: Sequence[float],
    expected_size: int,
) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if raw.shape != (expected_size,):
        raise ValueError(f"distribution must have shape ({expected_size},)")
    if not np.all(np.isfinite(raw)):
        raise ValueError("distribution must be finite")
    if np.any(raw < 0):
        raise ValueError("distribution must be nonnegative")
    total = float(raw.sum())
    if np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        return raw.copy()
    if total > 0 and np.all(np.equal(raw, np.rint(raw))):
        return raw / total
    raise ValueError("a probability distribution must sum to one")


def distribution_diagnostics(
    counts_or_probability: Sequence[float],
    constraints: MomentConstraints,
    *,
    benchmark: Sequence[float],
) -> dict[str, float | np.ndarray]:
    """Summarize a count vector or valid probability against a benchmark."""
    p = _counts_or_probability(counts_or_probability, constraints.size)
    q = _validate_probability(benchmark, name="benchmark")
    if q.shape != p.shape:
        raise ValueError("benchmark must have one value per outcome")

    outcomes = constraints.outcomes.astype(float)
    mean = float(outcomes @ p)
    variance = float(((outcomes - mean) ** 2) @ p)
    residuals = np.concatenate(
        ([float(p.sum() - 1.0)], constraints.features @ p - constraints.targets)
    )
    difference = p - q
    return {
        "probability_sum": float(p.sum()),
        "mean": mean,
        "variance": variance,
        "entropy": shannon_entropy(p),
        "maxent_entropy": shannon_entropy(q),
        "L1": float(np.linalg.norm(difference, ord=1)),
        "L2": float(np.linalg.norm(difference, ord=2)),
        "KL": kl_divergence(p, q),
        "constraint_residuals": residuals,
    }
