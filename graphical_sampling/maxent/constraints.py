"""Exact finite-discrete moment constraints and MaxEnt benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, root
from scipy.special import logsumexp


@dataclass(frozen=True)
class MomentConstraints:
    """Normalization plus integer-valued linear moment constraints."""

    outcomes: np.ndarray
    features: np.ndarray
    targets: np.ndarray

    @classmethod
    def from_moments(
        cls,
        outcomes: Sequence[int],
        features: Iterable[Sequence[int]],
        targets: Sequence[float],
    ) -> "MomentConstraints":
        outcome_array = np.asarray(outcomes)
        if outcome_array.ndim != 1 or len(outcome_array) == 0:
            raise ValueError("outcomes must be a nonempty one-dimensional sequence")
        if len(np.unique(outcome_array)) != len(outcome_array):
            raise ValueError("outcomes must be distinct")

        feature_rows = list(features)
        if feature_rows:
            feature_array = np.asarray(feature_rows)
            if feature_array.ndim != 2 or feature_array.shape[1] != len(outcome_array):
                raise ValueError("features must have one column per outcome")
        else:
            feature_array = np.empty((0, len(outcome_array)), dtype=np.int64)
        if not np.all(np.equal(feature_array, np.rint(feature_array))):
            raise ValueError("features must be integer-valued for exact signatures")

        target_array = np.asarray(targets, dtype=float)
        if target_array.ndim != 1 or len(target_array) != feature_array.shape[0]:
            raise ValueError("targets must contain one value per feature row")
        if not np.all(np.isfinite(target_array)):
            raise ValueError("targets must be finite")

        outcomes_int = np.asarray(outcome_array, dtype=np.int64)
        features_int = np.asarray(feature_array, dtype=np.int64)
        outcomes_int.setflags(write=False)
        features_int.setflags(write=False)
        target_array.setflags(write=False)
        return cls(outcomes_int, features_int, target_array)

    @property
    def size(self) -> int:
        return len(self.outcomes)

    @property
    def constraint_matrix(self) -> np.ndarray:
        return np.vstack([np.ones(self.size, dtype=np.int64), self.features])

    def integer_rhs(self, D: int) -> np.ndarray:
        if not isinstance(D, (int, np.integer)) or int(D) <= 0:
            raise ValueError("D must be a positive integer")
        totals = [int(D)]
        for target in self.targets:
            exact = Fraction(str(float(target))) * int(D)
            if exact.denominator != 1:
                raise ValueError(
                    f"D={D} does not give an integer total for target {target}"
                )
            totals.append(exact.numerator)
        return np.asarray(totals, dtype=np.int64)

    def signature(self, labels: Sequence[int]) -> tuple[int, ...]:
        if self.features.shape[0] == 0:
            return ()
        index = {int(value): i for i, value in enumerate(self.outcomes)}
        try:
            columns = [index[int(label)] for label in labels]
        except KeyError as exc:
            raise ValueError(f"unknown outcome label {exc.args[0]}") from exc
        return tuple(int(value) for value in self.features[:, columns].sum(axis=1))

    def validate_counts(self, counts: Sequence[int]) -> np.ndarray:
        raw = np.asarray(counts)
        if raw.shape != (self.size,):
            raise ValueError(f"counts must have shape ({self.size},)")
        if not np.all(np.equal(raw, np.rint(raw))):
            raise ValueError("counts must be integers")
        count_array = np.asarray(raw, dtype=np.int64)
        if np.any(count_array < 0):
            raise ValueError("counts must be nonnegative")
        D = int(count_array.sum())
        if D <= 0:
            raise ValueError("counts must have positive total")
        expected = self.integer_rhs(D)
        actual = self.constraint_matrix @ count_array
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"counts violate constraints: actual={actual.tolist()}, "
                f"expected={expected.tolist()}"
            )
        return count_array


def find_feasible_counts(
    constraints: MomentConstraints,
    D: int,
    objective: Sequence[float] | None = None,
) -> np.ndarray:
    """Find an exact nonnegative integer count vector with SciPy MILP."""
    rhs = constraints.integer_rhs(D)
    if objective is None:
        objective_array = np.zeros(constraints.size, dtype=float)
    else:
        objective_array = np.asarray(objective, dtype=float)
        if objective_array.shape != (constraints.size,):
            raise ValueError("objective must have one value per outcome")
    result = milp(
        c=objective_array,
        integrality=np.ones(constraints.size, dtype=int),
        bounds=Bounds(np.zeros(constraints.size), np.full(constraints.size, int(D))),
        constraints=LinearConstraint(constraints.constraint_matrix, rhs, rhs),
        options={"time_limit": 30.0},
    )
    if not result.success or result.x is None:
        raise ValueError(
            f"integer constraint system is infeasible for D={D}: {result.message}"
        )
    counts = np.rint(result.x).astype(np.int64)
    try:
        return constraints.validate_counts(counts)
    except ValueError as exc:
        raise ValueError(
            f"MILP returned no exact integer-feasible counts for D={D}"
        ) from exc


def solve_discrete_maxent(
    constraints: MomentConstraints,
    *,
    tolerance: float = 1e-11,
) -> np.ndarray:
    """Solve the finite exponential-family MaxEnt moment equations."""
    if constraints.features.shape[0] == 0:
        return np.full(constraints.size, 1.0 / constraints.size)

    features = constraints.features.astype(float)

    def probabilities(theta: np.ndarray) -> np.ndarray:
        logits = theta @ features
        return np.exp(logits - logsumexp(logits))

    def residual(theta: np.ndarray) -> np.ndarray:
        return features @ probabilities(theta) - constraints.targets

    solution = root(residual, np.zeros(features.shape[0], dtype=float))
    final_residual = residual(solution.x)
    if (
        not solution.success
        or not np.all(np.isfinite(solution.x))
        or np.max(np.abs(final_residual)) > tolerance
    ):
        raise RuntimeError(
            "MaxEnt moment solver failed to reach the requested target: "
            f"{solution.message}; residual={final_residual.tolist()}"
        )
    result = probabilities(solution.x)
    if np.any(result <= 0) or not np.isclose(result.sum(), 1.0, atol=tolerance):
        raise RuntimeError("MaxEnt solver returned an invalid probability vector")
    return result
