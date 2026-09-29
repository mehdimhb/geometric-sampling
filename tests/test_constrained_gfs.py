"""Focused tests for the constrained-GFS research prototype."""

from dataclasses import replace

import numpy as np
import pytest

from graphical_sampling.constrained import (
    ConstraintSystem,
    DesignDiagnostics,
    GroupCapacity,
    compute_fip,
    compute_sip,
    design_entropy,
    infeasible_mass,
    is_feasible,
    validate_probability_mass,
)


CONSTRAINTS = ConstraintSystem(
    population_size=10,
    sample_size=4,
    forbidden_pairs=frozenset(
        {frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})}
    ),
    group_capacities=(GroupCapacity(frozenset({6, 7, 8}), 2),),
)


def test_feasibility_checks_pairs_and_group_capacity():
    assert is_feasible((0, 2, 4, 6), CONSTRAINTS)
    assert not is_feasible((0, 1, 4, 6), CONSTRAINTS)
    assert not is_feasible(
        (0, 2, 6, 7, 8), replace(CONSTRAINTS, sample_size=5)
    )


def test_invalid_samples_are_rejected():
    with pytest.raises(ValueError, match="distinct"):
        is_feasible((0, 0, 2, 4), CONSTRAINTS)
    with pytest.raises(ValueError, match="range"):
        is_feasible((0, 2, 4, 10), CONSTRAINTS)
    with pytest.raises(ValueError, match="sample size"):
        is_feasible((0, 2, 4), CONSTRAINTS)


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: GroupCapacity(frozenset(), 0), "nonempty"),
        (lambda: GroupCapacity(frozenset({1, 2}), -1), "capacity"),
        (
            lambda: ConstraintSystem(
                population_size=3,
                sample_size=2,
                forbidden_pairs=frozenset({frozenset({0})}),
            ),
            "two distinct",
        ),
        (
            lambda: ConstraintSystem(
                population_size=3,
                sample_size=2,
                group_capacities=(GroupCapacity(frozenset({0, 3}), 1),),
            ),
            "range",
        ),
    ],
)
def test_invalid_constraints_are_rejected(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


SMALL_CONSTRAINTS = ConstraintSystem(
    population_size=4,
    sample_size=2,
    forbidden_pairs=frozenset({frozenset({0, 2})}),
)
SMALL_MASS = {(0, 2): 0.4, (1, 3): 0.6}
SMALL_FIP = np.array([0.4, 0.6, 0.4, 0.6])


def test_hand_computed_design_diagnostics():
    np.testing.assert_allclose(compute_fip(SMALL_MASS, 4), SMALL_FIP)
    sip = compute_sip(SMALL_MASS, 4)
    np.testing.assert_allclose(np.diag(sip), SMALL_FIP)
    assert sip[0, 2] == pytest.approx(0.4)
    assert sip[2, 0] == pytest.approx(0.4)
    assert design_entropy(SMALL_MASS) == pytest.approx(
        -(0.4 * np.log(0.4) + 0.6 * np.log(0.6))
    )
    assert infeasible_mass(SMALL_MASS, SMALL_CONSTRAINTS) == pytest.approx(0.4)


def test_probability_mass_validation_reports_infeasible_support_when_allowed():
    diagnostics = validate_probability_mass(
        SMALL_MASS,
        SMALL_FIP,
        SMALL_CONSTRAINTS,
        require_feasible=False,
    )
    assert isinstance(diagnostics, DesignDiagnostics)
    assert diagnostics.probability_sum_error == pytest.approx(0.0)
    assert diagnostics.max_fip_error == pytest.approx(0.0)
    assert diagnostics.support_size == 2
    assert diagnostics.violating_support_size == 1
    assert diagnostics.infeasible_mass == pytest.approx(0.4)
    np.testing.assert_allclose(diagnostics.fip, SMALL_FIP)
    np.testing.assert_allclose(np.diag(diagnostics.sip), SMALL_FIP)

    with pytest.raises(ValueError, match="hard constraints"):
        validate_probability_mass(
            SMALL_MASS,
            SMALL_FIP,
            SMALL_CONSTRAINTS,
            require_feasible=True,
        )


@pytest.mark.parametrize(
    "mass, prescribed_fip, message",
    [
        ({(0, 2): -0.1, (1, 3): 1.1}, SMALL_FIP, "nonnegative"),
        ({(0, 2): 0.4, (1, 3): 0.5}, SMALL_FIP, "sum to one"),
        (SMALL_MASS, [0.5, 0.5, 0.5, 0.5], "prescribed FIPs"),
        ({(0, 4): 0.4, (1, 3): 0.6}, SMALL_FIP, "range"),
        ({(0, 0): 0.4, (1, 3): 0.6}, SMALL_FIP, "distinct"),
    ],
)
def test_probability_mass_validation_rejects_broken_designs(
    mass, prescribed_fip, message
):
    with pytest.raises(ValueError, match=message):
        validate_probability_mass(
            mass,
            prescribed_fip,
            SMALL_CONSTRAINTS,
            require_feasible=False,
        )
