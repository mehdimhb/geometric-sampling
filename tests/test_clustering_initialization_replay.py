"""Initialization regressions using actual weighted, constrained and OT fits."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from graphical_sampling import Population
from graphical_sampling.clustering.fip_balanced_nmeans import FIPBalancedNMeans, OpenTSPSolver


X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
PI = np.full(4, .5)
# On the symmetric square, horizontal and vertical centroid pairs are fixed
# points of both constrained assignment and the entropic centroid iteration.
HORIZONTAL = np.array([[0., .5], [1., .5]])
VERTICAL = np.array([[.5, 0.], [.5, 1.]])


@pytest.mark.parametrize('method', ['weighted', 'ot', 'expanded'])
def test_supplied_centers_choose_the_requested_geometric_partition(method):
    outputs = []
    for centers in [HORIZONTAL, VERTICAL]:
        solver = FIPBalancedNMeans(2, init_clust_method=method, split_size=.25)
        labels, fitted = solver._get_labels_centroids(X, PI, centers)
        outputs.append(labels[:, None] == labels[None, :])
        # Labels may be permuted; the supplied centroid geometry is preserved.
        assert cdist(fitted, centers).min(axis=1).max() < 1e-8
    np.testing.assert_array_equal(outputs[0], [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])
    np.testing.assert_array_equal(outputs[1], [[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])


@pytest.mark.parametrize('method', ['weighted', 'ot', 'expanded'])
@pytest.mark.parametrize('centers', [np.zeros((3, 2)), np.zeros((2, 3)),
                                     np.array([[np.nan, 0.], [1., 1.]])])
def test_supplied_invalid_centers_never_fall_back_silently(method, centers):
    with pytest.raises(ValueError, match='init_centroids'):
        FIPBalancedNMeans(2, init_clust_method=method)._get_labels_centroids(X, PI, centers)


@pytest.mark.parametrize('method', ['weighted', 'expanded', 'ot'])
def test_existing_no_centers_initializers_still_work(method):
    labels, centers = FIPBalancedNMeans(2, n_init=2, init_clust_method=method,
                                      split_size=.25)._get_labels_centroids(X, PI)
    assert centers.shape == (2, 2) and np.isfinite(centers).all()
    assert len(np.unique(labels)) == 2


@pytest.mark.parametrize('n,r', [(4,1),(4,2)])
def test_constructed_cluster_count_and_grouped_sampling(n, r):
    x = np.column_stack([np.arange(8.), np.zeros(8)])
    pop = Population(x, np.full(8, .5))
    solver = FIPBalancedNMeans(n, r_sample_per_cluster=r, init_clust_method='weighted', n_init=2)
    centers = pop.coords[np.arange(0, 8, 2*r)]
    solver.fit(pop, init_centroids=centers)
    assert solver.K == n // r and len(solver.clusters) == n // r
    assert solver.membership.shape == (8, n // r)
    assert len(np.unique(solver.labels)) == n // r


@pytest.mark.parametrize('n,r', [(0,1),(3,2),(4,0),(4,1.5)])
def test_invalid_cluster_cardinality_is_rejected(n, r):
    with pytest.raises(ValueError):
        FIPBalancedNMeans(n, r_sample_per_cluster=r)


def test_fit_rejects_mismatched_population_target():
    pop = Population(X, PI)
    with pytest.raises(ValueError, match='n'):
        FIPBalancedNMeans(3).fit(pop)


def test_tsp_explicit_seed_replays_without_changing_numpy_global_rng():
    points = np.random.default_rng(120).random((12, 2))
    solver = OpenTSPSolver(random_state=19)
    np.random.seed(321); before = np.random.get_state()
    first = solver.solve(points)
    # An intervening solve must not affect the replay of a fresh per-call stream.
    OpenTSPSolver(random_state=25).solve(points)
    np.testing.assert_array_equal(first, solver.solve(points))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert sorted(first.tolist()) == list(range(12))


@pytest.mark.parametrize('method', ['weighted','ot','expanded'])
def test_seeded_clustering_replays_on_the_same_instance(method):
    x = np.random.default_rng(12).random((16, 2))
    pop = Population(x, np.full(16, .25))
    solver = FIPBalancedNMeans(4, init_clust_method=method, n_init=2,
                              split_size=.125, random_state=21)
    solver.fit(pop); first = (solver.path_order.copy(),solver.labels.copy(),solver.centroids.copy())
    solver.fit(pop)
    for a,b in zip(first,(solver.path_order,solver.labels,solver.centroids)):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('state', [-1, True, 1.2, 2**32])
def test_public_seed_validation(state):
    with pytest.raises(ValueError, match='random_state'):
        FIPBalancedNMeans(2, random_state=state)
    with pytest.raises(ValueError, match='random_state'):
        OpenTSPSolver(random_state=state)


def test_seeded_tsp_does_not_consume_numba_global_rng():
    from numba import njit

    @njit
    def seed_and_draw(seed):
        np.random.seed(seed)
        return np.random.random(3)

    @njit
    def draw():
        return np.random.random(3)

    seed_and_draw(88)
    expected = draw()
    seed_and_draw(88)
    OpenTSPSolver(random_state=19).solve(np.random.default_rng(4).random((12, 2)))
    np.testing.assert_array_equal(draw(), expected)


def test_seeded_exact_distance_ties_replay():
    points = np.array([[0., 0.], [0., 0.], [1., 0.], [1., 0.], [0., 1.]])
    solver = OpenTSPSolver(random_state=0)
    path = solver.solve(points)
    assert sorted(path.tolist()) == list(range(len(points)))
    np.testing.assert_array_equal(path, solver.solve(points))


@pytest.mark.parametrize('r,zones', [(1, 2), (2, 3)])
def test_recursive_zone_fitting_retains_fractional_mass_support(r, zones):
    # Subsets retain their parent's n, while zones have fractional target masses.
    pop = Population(np.column_stack([np.arange(8.), np.zeros(8)]), np.full(8, .5))
    solver = FIPBalancedNMeans(4, r_sample_per_cluster=r, init_clust_method='weighted', random_state=3)
    solver.fit(pop, init_centroids=pop.coords[::2*r])
    solver.fit_zones(zones, mode='cluster')
    assert len(solver.clusters) == 4 // r
    for cluster in solver.clusters:
        assert len(cluster.zones) == zones
        for zone in cluster.zones:
            assert np.sum(pop.inclusions[zone.indices] * zone.shares) == pytest.approx(r / zones)
