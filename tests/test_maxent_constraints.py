"""Exact constraints, integer feasibility, and discrete MaxEnt benchmarks."""
import numpy as np
import pytest

from graphical_sampling.maxent.constraints import (
    MomentConstraints,
    find_feasible_counts,
    solve_discrete_maxent,
)


X = np.arange(1, 7, dtype=int)
EXPECTED_MEAN_MAXENT = np.array([
    0.1030652452,
    0.1227305335,
    0.1461480427,
    0.1740337124,
    0.2072400869,
    0.2467823792,
])
EXPECTED_TWO_MOMENT_MAXENT = np.array([
    0.0410316102,
    0.1083937108,
    0.2007735633,
    0.2607509432,
    0.2374445294,
    0.1516056431,
])


def test_integer_rhs_signature_and_exact_count_validation():
    mean = MomentConstraints.from_moments(X, [X], [4.0])
    assert mean.integer_rhs(10_000).tolist() == [10_000, 40_000]
    assert mean.signature([2, 6]) == (8,)
    counts = np.array([0, 5_000, 0, 0, 0, 5_000])
    np.testing.assert_array_equal(mean.validate_counts(counts), counts)
    with pytest.raises(ValueError, match="constraints"):
        mean.validate_counts([1, 1, 1, 1, 1, 1])


def test_maxent_solver_matches_independent_die_benchmarks():
    unconstrained = MomentConstraints.from_moments(X, [], [])
    np.testing.assert_allclose(solve_discrete_maxent(unconstrained), np.full(6, 1 / 6), atol=0)

    mean = MomentConstraints.from_moments(X, [X], [4.0])
    np.testing.assert_allclose(solve_discrete_maxent(mean), EXPECTED_MEAN_MAXENT, atol=5e-9)

    both = MomentConstraints.from_moments(X, [X, X**2], [4.0, 17.8475])
    result = solve_discrete_maxent(both)
    np.testing.assert_allclose(result, EXPECTED_TWO_MOMENT_MAXENT, atol=5e-9)
    np.testing.assert_allclose(both.features @ result, both.targets, atol=1e-10)


def test_integer_feasibility_rejects_parity_obstruction_and_finds_d20000_counts():
    both = MomentConstraints.from_moments(X, [X, X**2], [4.0, 17.8475])
    with pytest.raises(ValueError, match="infeasible"):
        find_feasible_counts(both, 10_000, objective=X**3)

    counts = find_feasible_counts(both, 20_000, objective=X**3)
    assert np.issubdtype(counts.dtype, np.integer)
    np.testing.assert_array_equal(
        both.constraint_matrix @ counts,
        np.array([20_000, 80_000, 356_950]),
    )
    assert np.count_nonzero(counts) < 6


def test_unreachable_maxent_target_and_nonintegral_rhs_fail_loudly():
    impossible = MomentConstraints.from_moments(X, [X], [7.0])
    with pytest.raises(RuntimeError, match="MaxEnt"):
        solve_discrete_maxent(impossible)

    fractional = MomentConstraints.from_moments(X, [X], [4.00005])
    with pytest.raises(ValueError, match="integer"):
        fractional.integer_rhs(10_000)


def test_constraints_reject_noninteger_features_and_shape_mismatch():
    with pytest.raises(ValueError, match="integer-valued"):
        MomentConstraints.from_moments(X, [X / 2], [2.0])
    with pytest.raises(ValueError, match="targets"):
        MomentConstraints.from_moments(X, [X, X**2], [4.0])
