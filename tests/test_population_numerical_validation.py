"""Regression coverage for fixed-size inference; never normalize supplied pi silently."""
import numpy as np
import pytest
from graphical_sampling import Population


def coords(N):
    return np.column_stack([np.arange(N, dtype=float), np.zeros(N)])


@pytest.mark.parametrize('n', [4, 8, 16, 32])
def test_historical_normalization_does_not_lose_a_cluster(n):
    pi = n * np.ones(144) / 144
    pi *= n / pi.sum()
    before = pi.copy()
    pop = Population(coords(144), pi)
    assert pop.n == n
    assert isinstance(pop.n, int)
    np.testing.assert_array_equal(pi, before)
    np.testing.assert_array_equal(pop.inclusions, before)


@pytest.mark.parametrize('offset', [-1e-12, 0., 1e-12])
def test_integer_inference_accepts_only_roundoff_without_changing_pi(offset):
    pi = np.array([.5, .5, .5, .5 + offset])
    pop = Population(coords(4), pi)
    assert pop.n == 2
    np.testing.assert_array_equal(pop.inclusions, pi)


@pytest.mark.parametrize('pi', [np.full(4, .6), np.full(4, .1),
                                np.array([.5, .5, .5, .50001])])
def test_fractional_total_is_rejected_instead_of_truncated(pi):
    with pytest.raises(ValueError, match='integer'):
        Population(coords(4), pi)


@pytest.mark.parametrize('n', [0, -1, 5, 2.5, 2., True, np.nan, np.inf])
def test_invalid_explicit_target_is_rejected(n):
    with pytest.raises(ValueError, match='n'):
        Population(coords(4), np.ones(4), n=n)


@pytest.mark.parametrize('pi', [np.array([np.inf, 1.]), np.array([np.nan, 1.]),
                                np.array([0., 1.]), np.array([-.1, 1.])])
def test_invalid_weights_are_rejected_before_normalization(pi):
    with pytest.raises(ValueError):
        Population(coords(2), pi, n=1)


def test_explicit_integer_retains_weight_normalization_and_input():
    weights = np.array([1., 2., 3., 4.])
    pop = Population(coords(4), weights, n=np.int64(2))
    assert pop.n == 2
    np.testing.assert_allclose(pop.inclusions, [.2, .4, .6, .8], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(weights, [1., 2., 3., 4.])


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_nonfinite_coordinates_are_rejected(bad):
    x = coords(4); x[0, 1] = bad
    with pytest.raises(ValueError, match='finite'):
        Population(x, np.full(4, .5))
