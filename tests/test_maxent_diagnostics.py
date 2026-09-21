"""Diagnostics and public API for the graphical MaxEnt subsystem."""
import numpy as np
import pytest

from graphical_sampling.maxent import (
    ChainResult,
    DiscreteCountState,
    MomentConstraints,
    ReplacementKernel,
    distribution_diagnostics,
    find_feasible_counts,
    kl_divergence,
    run_chain,
    shannon_entropy,
    solve_discrete_maxent,
)


def test_hand_computed_entropy_kl_and_distribution_diagnostics():
    constraints = MomentConstraints.from_moments([1, 2], [], [])
    probability = np.array([0.5, 0.5])

    assert shannon_entropy(probability) == pytest.approx(np.log(2))
    assert kl_divergence(probability, probability) == pytest.approx(0.0)
    row = distribution_diagnostics([1, 1], constraints, benchmark=probability)
    assert row["probability_sum"] == pytest.approx(1.0)
    assert row["mean"] == pytest.approx(1.5)
    assert row["variance"] == pytest.approx(0.25)
    assert row["entropy"] == pytest.approx(np.log(2))
    assert row["maxent_entropy"] == pytest.approx(np.log(2))
    assert row["L1"] == pytest.approx(0.0)
    assert row["L2"] == pytest.approx(0.0)
    assert row["KL"] == pytest.approx(0.0)
    np.testing.assert_array_equal(row["constraint_residuals"], [0.0])


def test_kl_rejects_missing_benchmark_support_and_inputs_are_not_silently_fixed():
    with pytest.raises(ValueError, match="positive"):
        kl_divergence([1.0, 0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="sum to one"):
        shannon_entropy([0.4, 0.4])
    with pytest.raises(ValueError, match="nonnegative"):
        shannon_entropy([1.1, -0.1])


def test_diagnostics_report_literal_constraint_residuals():
    outcomes = np.arange(1, 7)
    constraints = MomentConstraints.from_moments(outcomes, [outcomes], [4.0])
    benchmark = solve_discrete_maxent(constraints)
    row = distribution_diagnostics(
        [0.1, 0.1, 0.1, 0.2, 0.2, 0.3],
        constraints,
        benchmark=benchmark,
    )
    np.testing.assert_allclose(row["constraint_residuals"], [0.0, 0.2])
    assert row["mean"] == pytest.approx(4.2)


def test_public_api_exports_reusable_types_and_functions():
    assert MomentConstraints is not None
    assert DiscreteCountState is not None
    assert ReplacementKernel is not None
    assert ChainResult is not None
    assert callable(find_feasible_counts)
    assert callable(solve_discrete_maxent)
    assert callable(run_chain)
