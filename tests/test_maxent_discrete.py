"""Exact count projection and reversibility of the atom-block chain."""
from itertools import product
from math import factorial

import numpy as np
import pytest

from graphical_sampling.maxent.constraints import MomentConstraints, find_feasible_counts
from graphical_sampling.maxent.discrete import (
    DiscreteCountState,
    ReplacementKernel,
    enumerate_count_transitions,
    run_chain,
)


def normalization_constraints(outcomes):
    return MomentConstraints.from_moments(outcomes, [], [])


def multinomial_weight(counts):
    result = factorial(sum(counts))
    for count in counts:
        result //= factorial(count)
    return result


def test_one_atom_transition_uses_source_multiplicity_and_keeps_self_loops():
    state = DiscreteCountState([2, 1], normalization_constraints([1, 2]))
    kernel = ReplacementKernel(state.constraints, {1: 1.0})
    transitions = enumerate_count_transitions(state.counts, kernel, 1)

    # Choose label 1 with probability 2/3, then replacement label 2 with 1/2.
    assert transitions[(1, 2)] == pytest.approx((2 / 3) * (1 / 2))
    assert transitions[(2, 1)] > 0
    assert sum(transitions.values()) == pytest.approx(1.0)


def test_count_chain_obeys_multinomial_detailed_balance():
    constraints = normalization_constraints([1, 2])
    kernel = ReplacementKernel(constraints, {1: 1.0})
    states = [(left, 3 - left) for left in range(4)]
    transition_rows = {
        state: enumerate_count_transitions(state, kernel, 1) for state in states
    }

    for source, target in product(states, repeat=2):
        forward = transition_rows[source].get(target, 0.0)
        reverse = transition_rows[target].get(source, 0.0)
        assert multinomial_weight(source) * forward == pytest.approx(
            multinomial_weight(target) * reverse,
            abs=1e-14,
        )


def test_signature_groups_retain_source_patterns_and_cap_large_tables():
    outcomes = np.arange(1, 7)
    mean = MomentConstraints.from_moments(outcomes, [outcomes], [4.0])
    kernel = ReplacementKernel(mean, {2: 1.0})
    assert kernel.contains_pattern(2, (8,), (2, 6))

    with pytest.raises(ValueError, match="pattern cap"):
        ReplacementKernel(mean, {12: 1.0}, max_patterns=100_000)


def test_chain_preserves_mean_and_two_moments_exactly():
    outcomes = np.arange(1, 7)
    mean = MomentConstraints.from_moments(outcomes, [outcomes], [4.0])
    mean_result = run_chain(
        [0, 500, 0, 0, 0, 500],
        mean,
        ReplacementKernel(mean, {2: 1.0}),
        iterations=500,
        seed=7,
        checkpoint_every=100,
    )
    for counts in mean_result.checkpoint_counts:
        mean.validate_counts(counts)
    assert mean_result.changes > 0

    both = MomentConstraints.from_moments(
        outcomes,
        [outcomes, outcomes**2],
        [4.0, 17.8475],
    )
    initial = find_feasible_counts(both, 800, objective=outcomes**3)
    both_result = run_chain(
        initial,
        both,
        ReplacementKernel(both, {3: 0.1, 4: 0.2, 5: 0.3, 6: 0.4}),
        iterations=500,
        seed=8,
        checkpoint_every=100,
    )
    for counts in both_result.checkpoint_counts:
        both.validate_counts(counts)
    assert both_result.changes > 0


def test_seed_replay_is_exact_and_checkpoints_include_zero_and_final():
    constraints = normalization_constraints([1, 2, 3])
    kernel = ReplacementKernel(constraints, {1: 1.0})
    kwargs = dict(
        initial_counts=[2, 3, 4],
        constraints=constraints,
        kernel=kernel,
        iterations=11,
        seed=123,
        checkpoint_every=5,
        burn_in=5,
    )
    first = run_chain(**kwargs)
    second = run_chain(**kwargs)

    np.testing.assert_array_equal(first.final_counts, second.final_counts)
    np.testing.assert_array_equal(first.checkpoint_counts, second.checkpoint_counts)
    np.testing.assert_array_equal(first.checkpoint_iterations, [0, 5, 10, 11])
    assert first.proposals == 11
    assert first.burn_in == 5
    assert first.change_rate == pytest.approx(first.changes / 11)


def test_tracked_zero_step_chain_returns_actual_ordered_atoms():
    constraints = normalization_constraints([1, 2, 3])
    result = run_chain(
        [2, 1, 3],
        constraints,
        ReplacementKernel(constraints, {1: 1.0}),
        iterations=0,
        seed=41,
        checkpoint_every=2,
        track_atoms=True,
    )

    expected = np.array([1, 1, 2, 3, 3, 3])
    np.testing.assert_array_equal(result.initial_atoms, expected)
    np.testing.assert_array_equal(result.final_atoms, expected)
    np.testing.assert_array_equal(result.final_counts, [2, 1, 3])
    np.testing.assert_array_equal(result.checkpoint_iterations, [0])


def test_count_only_chain_does_not_allocate_atom_endpoints():
    constraints = normalization_constraints([1, 2])
    result = run_chain(
        [2, 2],
        constraints,
        ReplacementKernel(constraints, {1: 1.0}),
        iterations=2,
        seed=42,
        checkpoint_every=1,
    )
    assert result.initial_atoms is None
    assert result.final_atoms is None


def test_tracked_chain_preserves_moments_and_atom_multiplicities():
    outcomes = np.arange(1, 7)
    constraints = MomentConstraints.from_moments(outcomes, [outcomes], [4.0])
    kwargs = dict(
        initial_counts=[0, 100, 0, 0, 0, 100],
        constraints=constraints,
        kernel=ReplacementKernel(constraints, {2: 1.0}),
        iterations=300,
        seed=43,
        checkpoint_every=37,
        track_atoms=True,
    )
    first = run_chain(**kwargs)
    second = run_chain(**kwargs)

    np.testing.assert_array_equal(first.final_atoms, second.final_atoms)
    np.testing.assert_array_equal(first.final_counts, second.final_counts)
    derived = np.array(
        [np.count_nonzero(first.final_atoms == value) for value in outcomes]
    )
    np.testing.assert_array_equal(derived, first.final_counts)
    for counts in first.checkpoint_counts:
        constraints.validate_counts(counts)


def test_atom_step_selects_distinct_positions_when_labels_repeat():
    class RecordingRng:
        def __init__(self):
            self.base = np.random.default_rng(44)
            self.positions = None
            self.replace = None

        def random(self):
            return self.base.random()

        def choice(self, values, *, size, replace):
            self.replace = replace
            self.positions = self.base.choice(values, size=size, replace=replace)
            return self.positions

        def integers(self, *args, **kwargs):
            return self.base.integers(*args, **kwargs)

    constraints = normalization_constraints([1, 2])
    kernel = ReplacementKernel(constraints, {2: 1.0})
    atoms = np.array([1, 1], dtype=np.int64)
    counts = np.array([2, 0], dtype=np.int64)
    rng = RecordingRng()
    result = kernel.step_atoms_inplace(atoms, counts, rng)
    assert result.source_labels == (1, 1)
    assert rng.replace is False
    assert len(np.unique(rng.positions)) == 2
    assert atoms.shape == (2,)
    assert counts.sum() == 2
