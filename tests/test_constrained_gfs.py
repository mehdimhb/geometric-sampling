"""Focused tests for the constrained-GFS research prototype."""

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

import graphical_sampling.constrained as constrained_module
from graphical_sampling import Design
from graphical_sampling.constrained import (
    ConstraintSystem,
    DesignDiagnostics,
    GroupCapacity,
    _Float64Design,
    analyze_pair_attainability,
    attainable_pair_score,
    build_madow_design,
    compute_fip,
    compute_sip,
    design_entropy,
    enumerate_feasible_samples,
    incidence_matrix,
    infeasible_mass,
    is_feasible,
    propose_gfs_exchange,
    role_rng,
    run_constraint_repair,
    run_feasible_gfs_search,
    seeded_madow_order,
    solve_feasibility_reference,
    solve_numerical_maxent,
    validate_probability_mass,
)
from graphical_sampling.maxent.finite_population import design_probability_mass
from graphical_sampling.structs import _Sample


CONSTRAINTS = ConstraintSystem(
    population_size=10,
    sample_size=4,
    forbidden_pairs=frozenset(
        {frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})}
    ),
    group_capacities=(GroupCapacity(frozenset({6, 7, 8}), 2),),
)
PI = np.array([0.55, 0.45, 0.50, 0.40, 0.35, 0.35, 0.40, 0.35, 0.35, 0.30])
COORDINATES = np.column_stack(
    [np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64) % 3]
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


def test_role_zero_generates_the_predeclared_madow_order():
    np.testing.assert_array_equal(
        seeded_madow_order(10, 12345),
        [4, 8, 1, 3, 7, 9, 6, 0, 2, 5],
    )


def test_role_streams_are_replayable_and_isolated():
    entropy_before = role_rng(12345, 2).random(8)
    role_rng(12345, 1).random(10_000)
    np.testing.assert_array_equal(role_rng(12345, 2).random(8), entropy_before)
    assert not np.array_equal(
        role_rng(12345, 2).random(8), role_rng(12345, 3).random(8)
    )


@pytest.mark.parametrize(
    "bad_fip",
    [
        np.array([[0.5, 0.5]]),
        np.array([0.5, np.nan, 0.5]),
        np.array([0.5, np.inf, 0.5]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.5, -0.1, 0.6]),
        np.array([1.1, 0.4, 0.5]),
        np.array([0.4, 0.4, 0.4]),
    ],
)
def test_invalid_fips_fail_before_population_construction(monkeypatch, bad_fip):
    class UnexpectedPopulation:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Population construction must not be reached")

    monkeypatch.setattr(constrained_module, "Population", UnexpectedPopulation)
    coordinates = np.zeros((bad_fip.size, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="prescribed_fip"):
        build_madow_design(bad_fip, coordinates, np.arange(bad_fip.size))


def test_seeded_madow_design_preserves_fips_and_starts_with_violations():
    design = build_madow_design(
        PI,
        COORDINATES,
        seeded_madow_order(10, 12345),
    )
    assert isinstance(design, _Float64Design)
    assert _Float64Design.exchange is Design.exchange
    assert _Float64Design._switch is Design._switch

    mass = design_probability_mass(design)
    assert all(len(sample) == 4 for sample in mass)
    np.testing.assert_allclose(compute_fip(mass, 10), PI, atol=1e-12, rtol=0.0)
    assert infeasible_mass(mass, CONSTRAINTS) == pytest.approx(0.9, abs=1e-12)


def make_production_madow_design():
    return build_madow_design(
        PI,
        COORDINATES,
        seeded_madow_order(10, 12345),
    )


def test_proposal_calls_inherited_exchange_without_mutating_input(monkeypatch):
    design = make_production_madow_design()
    before_mass = design_probability_mass(design)
    before_heaps = [tuple(heap.heap) for heap in design._heaps]
    calls = []
    original = Design.exchange

    def counted_exchange(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Design, "exchange", counted_exchange)
    proposal = propose_gfs_exchange(
        design,
        PI,
        CONSTRAINTS,
        rng=role_rng(12345, 1),
        exchange_coef=0.5,
    )

    assert calls == [((), {"partitions": 1, "pull_strategy": "random", "exchange_coef": 0.5})]
    assert design_probability_mass(design) == before_mass
    assert [tuple(heap.heap) for heap in design._heaps] == before_heaps
    assert proposal.candidate is not None
    assert proposal.diagnostics is not None
    assert not proposal.no_op


def test_inherited_full_exchange_can_be_an_aggregated_structural_no_op():
    design = make_production_madow_design()
    first = _Sample(0.3, frozenset({1, 2, 3, 4}))
    second = _Sample(0.3, frozenset({1, 2, 3, 5}))
    pieces = design._switch(first, second, 1.0, zone_idx=0)
    aggregated = {}
    for piece in pieces:
        aggregated[piece.ids] = aggregated.get(piece.ids, 0.0) + piece.prob
    assert aggregated == pytest.approx({first.ids: 0.3, second.ids: 0.3})


def test_proposal_rejects_unchanged_aggregate_as_structural_no_op(monkeypatch):
    design = make_production_madow_design()
    monkeypatch.setattr(Design, "exchange", lambda self, *args, **kwargs: None)
    proposal = propose_gfs_exchange(
        design,
        PI,
        CONSTRAINTS,
        rng=role_rng(12345, 1),
        exchange_coef=1.0,
    )
    assert proposal.no_op is True
    assert proposal.candidate is None
    assert proposal.diagnostics is None


def test_repair_reaches_feasibility_through_strict_gfs_improvements():
    result = run_constraint_repair(
        make_production_madow_design(),
        PI,
        CONSTRAINTS,
        proposals=2_000,
        seed=np.random.SeedSequence([12345, 1]),
        checkpoint_every=250,
    )
    assert result.reached_feasibility
    assert result.status == "feasible"
    assert result.trace[0].infeasible_mass == pytest.approx(0.9, abs=1e-12)
    assert result.trace[-1].infeasible_mass <= 1e-12
    assert result.counts.accepted > 0
    assert result.counts.proposed <= 2_000
    assert result.counts.invariant_failures == 0

    final_mass = design_probability_mass(result.final_design)
    validate_probability_mass(
        final_mass,
        PI,
        CONSTRAINTS,
        require_feasible=True,
        tolerance=1e-10,
    )

    accepted_rows = []
    previous_accepted = 0
    for row in result.trace:
        if row.accepted > previous_accepted:
            assert row.accepted == previous_accepted + 1
            accepted_rows.append(row)
            previous_accepted = row.accepted
    accepted_v = np.array(
        [result.trace[0].infeasible_mass]
        + [row.infeasible_mass for row in accepted_rows]
    )
    assert np.all(np.diff(accepted_v) < -1e-12)
    assert any(row.infeasible_mass > 1e-12 for row in accepted_rows[:-1])


def test_repair_returns_immediately_for_an_initially_feasible_design():
    unconstrained = ConstraintSystem(population_size=10, sample_size=4)
    initial = make_production_madow_design()
    before = design_probability_mass(initial)
    result = run_constraint_repair(
        initial,
        PI,
        unconstrained,
        proposals=10,
        seed=np.random.SeedSequence([12345, 1]),
        checkpoint_every=100,
    )
    assert result.status == "already_feasible"
    assert result.reached_feasibility
    assert result.counts.proposed == 0
    assert result.counts.accepted == 0
    assert len(result.trace) == 1
    assert result.trace[0].proposal == 0
    assert design_probability_mass(result.final_design) == before


def test_zero_budget_repair_records_iteration_zero_without_changing_design():
    initial = make_production_madow_design()
    before = design_probability_mass(initial)
    result = run_constraint_repair(
        initial,
        PI,
        CONSTRAINTS,
        proposals=0,
        seed=np.random.SeedSequence([12345, 1]),
        checkpoint_every=100,
    )
    assert result.status == "budget_exhausted"
    assert not result.reached_feasibility
    assert result.counts.proposed == 0
    assert len(result.trace) == 1
    assert result.trace[0].proposal == 0
    assert design_probability_mass(result.final_design) == before


def test_repair_records_final_attempt_when_checkpoint_exceeds_budget():
    result = run_constraint_repair(
        make_production_madow_design(),
        PI,
        CONSTRAINTS,
        proposals=3,
        seed=np.random.SeedSequence([12345, 1]),
        checkpoint_every=100,
    )
    assert result.trace[-1].proposal == result.counts.proposed == 3


def test_repair_replays_exactly_and_balances_all_proposal_counts():
    kwargs = dict(
        prescribed_fip=PI,
        constraints=CONSTRAINTS,
        proposals=2_000,
        seed=np.random.SeedSequence([12345, 1]),
        checkpoint_every=250,
    )
    first = run_constraint_repair(make_production_madow_design(), **kwargs)
    kwargs["seed"] = np.random.SeedSequence([12345, 1])
    second = run_constraint_repair(make_production_madow_design(), **kwargs)

    assert design_probability_mass(first.final_design) == pytest.approx(
        design_probability_mass(second.final_design), abs=0.0, rel=0.0
    )
    assert first.counts == second.counts
    assert first.status == second.status
    stable_first = [
        (
            row.proposal,
            row.accepted,
            row.objective,
            row.infeasible_mass,
            row.entropy,
            row.support_size,
            row.max_fip_error,
        )
        for row in first.trace
    ]
    stable_second = [
        (
            row.proposal,
            row.accepted,
            row.objective,
            row.infeasible_mass,
            row.entropy,
            row.support_size,
            row.max_fip_error,
        )
        for row in second.trace
    ]
    assert stable_first == stable_second

    counts = first.counts
    assert counts.proposed == (
        counts.no_op
        + counts.invariant_failures
        + counts.feasibility_rejected
        + counts.accepted
        + counts.non_improving
    )
    assert counts.structurally_valid == (
        counts.accepted + counts.non_improving + counts.feasibility_rejected
    )
    assert counts.structurally_valid == (
        counts.constraint_feasible_candidates
        + counts.constraint_infeasible_candidates
    )
    assert counts.feasibility_rejected == 0
    assert counts.invariant_failures == 0


def test_enumeration_and_feasibility_reference_for_production_problem():
    samples = enumerate_feasible_samples(CONSTRAINTS)
    assert len(samples) == 122
    assert samples == tuple(sorted(samples))
    assert all(is_feasible(sample, CONSTRAINTS) for sample in samples)

    incidence = incidence_matrix(samples, 10)
    assert incidence.shape == (10, 122)
    np.testing.assert_array_equal(incidence.sum(axis=0), np.full(122, 4.0))
    result = solve_feasibility_reference(PI, samples, incidence)
    assert result.success
    assert result.status == "feasible"
    assert result.feasible_sample_count == 122
    assert result.normalization_error < 1e-10
    assert result.max_fip_error < 1e-10
    assert result.max_equality_error < 1e-10
    assert not hasattr(result, "probabilities")


def test_empty_feasible_space_is_reported_as_mathematical_infeasibility():
    constraints = ConstraintSystem(
        population_size=3,
        sample_size=2,
        forbidden_pairs=frozenset(
            {
                frozenset({0, 1}),
                frozenset({0, 2}),
                frozenset({1, 2}),
            }
        ),
    )
    samples = enumerate_feasible_samples(constraints)
    assert samples == ()
    incidence = incidence_matrix(samples, 3)
    assert incidence.shape == (3, 0)
    result = solve_feasibility_reference([2 / 3] * 3, samples, incidence)
    assert result.success is False
    assert result.status == "infeasible"
    assert result.feasible_sample_count == 0
    assert not hasattr(result, "probabilities")


def test_nonempty_but_fip_incompatible_space_is_explicitly_infeasible():
    constraints = ConstraintSystem(
        population_size=3,
        sample_size=2,
        forbidden_pairs=frozenset({frozenset({0, 1})}),
    )
    samples = enumerate_feasible_samples(constraints)
    incidence = incidence_matrix(samples, 3)
    assert samples
    result = solve_feasibility_reference([0.9, 0.9, 0.2], samples, incidence)
    assert result.success is False
    assert result.status == "infeasible"
    assert not hasattr(result, "probabilities")


def test_pair_attainability_identifies_explicit_forbidden_pairs():
    samples = enumerate_feasible_samples(CONSTRAINTS)
    incidence = incidence_matrix(samples, 10)
    result = analyze_pair_attainability(PI, samples, incidence, CONSTRAINTS)
    explicit = {(0, 1), (2, 3), (4, 5)}
    assert set(result.explicit_forbidden) == explicit
    for pair in explicit:
        assert result.categories[pair] == "explicit_forbidden"
        assert result.maximum_sip[pair] == pytest.approx(0.0)
    assert not (
        set(result.marginal_attainable)
        & (
            set(result.explicit_forbidden)
            | set(result.other_constraint_incompatible)
            | set(result.marginal_inattainable)
        )
    )
    categorized = (
        set(result.explicit_forbidden)
        | set(result.other_constraint_incompatible)
        | set(result.marginal_inattainable)
        | set(result.marginal_attainable)
    )
    assert categorized == set(result.categories) == set(combinations(range(10), 2))


def test_pair_attainability_separates_geometric_from_marginal_possibility():
    constraints = ConstraintSystem(population_size=3, sample_size=2)
    samples = enumerate_feasible_samples(constraints)
    incidence = incidence_matrix(samples, 3)
    assert samples == ((0, 1), (0, 2), (1, 2))
    result = analyze_pair_attainability(
        [1.0, 1.0, 0.0], samples, incidence, constraints
    )
    assert result.categories[(0, 1)] == "marginal_attainable"
    assert result.maximum_sip[0, 1] == pytest.approx(1.0)
    assert result.categories[(0, 2)] == "marginal_inattainable"
    assert result.categories[(1, 2)] == "marginal_inattainable"
    assert set(result.marginal_attainable) == {(0, 1)}
    assert set(result.marginal_inattainable) == {(0, 2), (1, 2)}


def test_numerical_maxent_reference_is_feasible_and_labeled_as_a_benchmark():
    samples = enumerate_feasible_samples(CONSTRAINTS)
    incidence = incidence_matrix(samples, 10)
    result = solve_numerical_maxent(PI, samples, incidence)
    assert result.success
    assert result.label == "Numerical constrained MaxEnt benchmark"
    assert "Numerical" in result.label
    assert "Exact" not in result.label
    assert result.probabilities.shape == (122,)
    assert np.all(result.probabilities >= 0.0)
    assert float(result.probabilities.sum()) == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(
        incidence @ result.probabilities, PI, atol=1e-9, rtol=0.0
    )
    assert result.max_equality_error < 1e-9
    assert np.isfinite(result.entropy)
    assert result.entropy <= np.log(122) + 1e-12


def test_attainable_pair_score_uses_only_the_supplied_pair_mask():
    pairs = ((0, 1), (1, 2))
    first_mass = {(0, 1): 0.4, (0, 2): 0.6}
    second_mass = {(0, 1): 0.4, (0, 3): 0.6}
    expected = np.log(0.4 + 1e-12) + np.log(1e-12)
    assert attainable_pair_score(first_mass, pairs) == pytest.approx(expected)
    assert attainable_pair_score(second_mass, pairs) == pytest.approx(expected)


@pytest.mark.parametrize("objective", ["entropy", "sip"])
def test_stage2_rejects_an_infeasible_initial_design(objective):
    with pytest.raises(ValueError, match="feasible initial design"):
        run_feasible_gfs_search(
            make_production_madow_design(),
            PI,
            CONSTRAINTS,
            objective=objective,
            attainable_pairs=(),
            proposals=1,
            seed=np.random.SeedSequence([12345, 2]),
        )


def make_repaired_production_design():
    result = run_constraint_repair(
        make_production_madow_design(),
        PI,
        CONSTRAINTS,
        proposals=2_000,
        seed=np.random.SeedSequence([12345, 1]),
        checkpoint_every=250,
    )
    assert result.reached_feasibility
    return result.final_design


def assert_stage2_accounting(result):
    counts = result.counts
    assert counts.proposed == (
        counts.no_op
        + counts.invariant_failures
        + counts.feasibility_rejected
        + counts.accepted
        + counts.non_improving
    )
    assert counts.structurally_valid == (
        counts.feasibility_rejected + counts.accepted + counts.non_improving
    )
    assert counts.structurally_valid == (
        counts.constraint_feasible_candidates
        + counts.constraint_infeasible_candidates
    )
    assert counts.constraint_infeasible_candidates == counts.feasibility_rejected
    assert counts.invariant_failures == 0


def accepted_objectives(result):
    values = [result.trace[0].objective]
    previous = 0
    for row in result.trace:
        if row.accepted > previous:
            assert row.accepted == previous + 1
            values.append(row.objective)
            previous = row.accepted
    return np.asarray(values)


def test_entropy_search_improves_feasibly_and_replays_exactly():
    repaired = make_repaired_production_design()
    initial_mass = design_probability_mass(repaired)
    initial_entropy = design_entropy(initial_mass)
    kwargs = dict(
        prescribed_fip=PI,
        constraints=CONSTRAINTS,
        objective="entropy",
        proposals=2_000,
        seed=np.random.SeedSequence([12345, 2]),
        checkpoint_every=250,
    )
    result = run_feasible_gfs_search(repaired, **kwargs)
    kwargs["seed"] = np.random.SeedSequence([12345, 2])
    replay = run_feasible_gfs_search(repaired, **kwargs)

    final_mass = design_probability_mass(result.final_design)
    diagnostics = validate_probability_mass(
        final_mass, PI, CONSTRAINTS, require_feasible=True, tolerance=1e-10
    )
    assert diagnostics.max_fip_error < 1e-10
    assert result.counts.accepted > 0
    assert result.counts.invariant_failures == 0
    assert diagnostics.entropy >= initial_entropy
    assert np.all(np.diff(accepted_objectives(result)) > 1e-12)
    assert_stage2_accounting(result)
    assert design_probability_mass(replay.final_design) == pytest.approx(
        final_mass, abs=0.0, rel=0.0
    )
    assert replay.counts == result.counts

    samples = enumerate_feasible_samples(CONSTRAINTS)
    reference = solve_numerical_maxent(PI, samples, incidence_matrix(samples, 10))
    assert reference.entropy >= diagnostics.entropy - 1e-9


def test_sip_search_improves_only_attainable_pairs_and_replays_exactly():
    repaired = make_repaired_production_design()
    samples = enumerate_feasible_samples(CONSTRAINTS)
    incidence = incidence_matrix(samples, 10)
    pair_analysis = analyze_pair_attainability(PI, samples, incidence, CONSTRAINTS)
    attainable = tuple(sorted(pair_analysis.marginal_attainable))
    initial_mass = design_probability_mass(repaired)
    initial_score = attainable_pair_score(initial_mass, attainable)
    kwargs = dict(
        prescribed_fip=PI,
        constraints=CONSTRAINTS,
        objective="sip",
        attainable_pairs=attainable,
        proposals=2_000,
        seed=np.random.SeedSequence([12345, 3]),
        checkpoint_every=250,
    )
    result = run_feasible_gfs_search(repaired, **kwargs)
    kwargs["seed"] = np.random.SeedSequence([12345, 3])
    replay = run_feasible_gfs_search(repaired, **kwargs)

    final_mass = design_probability_mass(result.final_design)
    diagnostics = validate_probability_mass(
        final_mass, PI, CONSTRAINTS, require_feasible=True, tolerance=1e-10
    )
    final_score = attainable_pair_score(final_mass, attainable)
    assert diagnostics.max_fip_error < 1e-10
    assert result.counts.accepted > 0
    assert final_score >= initial_score
    assert np.all(np.diff(accepted_objectives(result)) > 1e-12)
    assert_stage2_accounting(result)
    assert design_probability_mass(replay.final_design) == pytest.approx(
        final_mass, abs=0.0, rel=0.0
    )
    assert replay.counts == result.counts
    for pair in (
        set(pair_analysis.explicit_forbidden)
        | set(pair_analysis.other_constraint_incompatible)
    ):
        assert diagnostics.sip[pair] == pytest.approx(0.0, abs=1e-12)


def test_stage2_zero_budget_and_final_checkpoint_contracts():
    repaired = make_repaired_production_design()
    before = design_probability_mass(repaired)
    zero = run_feasible_gfs_search(
        repaired,
        PI,
        CONSTRAINTS,
        objective="entropy",
        proposals=0,
        seed=np.random.SeedSequence([12345, 2]),
        checkpoint_every=100,
    )
    assert len(zero.trace) == 1
    assert zero.trace[0].proposal == 0
    assert zero.counts.proposed == 0
    assert design_probability_mass(zero.final_design) == before

    partial = run_feasible_gfs_search(
        repaired,
        PI,
        CONSTRAINTS,
        objective="entropy",
        proposals=7,
        seed=np.random.SeedSequence([12345, 2]),
        checkpoint_every=5,
    )
    assert partial.trace[-1].proposal == partial.counts.proposed == 7


def test_role_two_consumption_cannot_change_a_fresh_role_three_replay():
    repaired = make_repaired_production_design()
    samples = enumerate_feasible_samples(CONSTRAINTS)
    pair_analysis = analyze_pair_attainability(
        PI, samples, incidence_matrix(samples, 10), CONSTRAINTS
    )
    settings = dict(
        prescribed_fip=PI,
        constraints=CONSTRAINTS,
        objective="sip",
        attainable_pairs=tuple(sorted(pair_analysis.marginal_attainable)),
        proposals=25,
        checkpoint_every=100,
    )
    first = run_feasible_gfs_search(
        repaired, seed=np.random.SeedSequence([12345, 3]), **settings
    )
    run_feasible_gfs_search(
        repaired,
        PI,
        CONSTRAINTS,
        objective="entropy",
        proposals=100,
        seed=np.random.SeedSequence([12345, 2]),
        checkpoint_every=100,
    )
    second = run_feasible_gfs_search(
        repaired, seed=np.random.SeedSequence([12345, 3]), **settings
    )
    assert first.counts == second.counts
    assert design_probability_mass(first.final_design) == pytest.approx(
        design_probability_mass(second.final_design), abs=0.0, rel=0.0
    )
