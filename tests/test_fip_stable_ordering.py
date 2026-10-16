"""Ordering contracts independent of clustering initialization and TSP search."""
import numpy as np
import pytest

from graphical_sampling.clustering.fip_balanced_nmeans import (
    FIPBalancedNMeans,
    OpenTSPSolver,
)


def ordered_units(radii):
    # Cluster zero lies on a ray from the singleton next cluster at the origin.
    # Its within-cluster scores are exactly -radii. Scrambled global IDs ensure
    # the tie rule respects input positions rather than sorted identifiers.
    coords = np.column_stack((np.r_[radii, 0.], np.zeros(len(radii) + 1)))
    labels = np.r_[np.zeros(len(radii), dtype=int), 1]
    identifiers = np.random.default_rng(18).permutation(len(coords)) + 100
    solver = FIPBalancedNMeans(4, r_sample_per_cluster=2)
    clusters = solver._generate_exact_clusters(
        np.array([0, 1]), labels, coords,
        np.full(len(coords), 4 / len(coords)), identifiers,
    )
    assert len(clusters) == 2
    return solver._before_indices


def test_equal_scores_retain_original_input_order():
    radii = np.tile([1., 3., 1., 2., 3., 2., 1., 3.], 4)
    # This fixture has mixed ranks and ties across quicksort partition boundaries.
    # State the expected order with Python's stable ordering, not NumPy's sort.
    expected = sorted(range(len(radii)), key=lambda i: -radii[i])
    np.testing.assert_array_equal(ordered_units(radii), expected + [len(radii)])


def test_untied_scores_keep_the_existing_distance_order():
    radii = np.random.default_rng(6).permutation(np.arange(1., 33.))
    expected = np.r_[np.argsort(-radii), len(radii)]
    np.testing.assert_array_equal(ordered_units(radii), expected)


def test_single_cluster_equal_scores_retain_input_order():
    count = 40
    solver = FIPBalancedNMeans(1)
    solver._generate_exact_clusters(
        np.array([0]), np.zeros(count, dtype=int),
        np.random.default_rng(8).random((count, 2)), np.full(count, 1/count),
        np.arange(count),
    )
    np.testing.assert_array_equal(solver._before_indices, np.arange(count))


@pytest.mark.parametrize('restarts', [1, 15])
def test_seeded_otsp_replays_after_an_intervening_solve(restarts):
    points = np.random.default_rng(120).random((16, 2))
    solver = OpenTSPSolver(random_state=19)
    first = solver.solve(points, restarts=restarts)
    OpenTSPSolver(random_state=25).solve(points, restarts=restarts)
    np.testing.assert_array_equal(first, solver.solve(points, restarts=restarts))
    np.testing.assert_array_equal(np.sort(first), np.arange(len(points)))


def test_default_otsp_restart_count_still_behaves_as_fifteen():
    points = np.random.default_rng(22).random((16, 2))
    solver = OpenTSPSolver(random_state=3)
    np.testing.assert_array_equal(solver.solve(points), solver.solve(points, restarts=15))
