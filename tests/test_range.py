import unittest
from geometric_sampling.structs import Range


class RangeTestCase(unittest.TestCase):
    def test_almost_zero(self):
        r = Range(1e-10, frozenset({1, 2, 3}))
        assert r.almost_zero()

    def test_compare(self):
        r1 = Range(0.3, frozenset({1, 2}))
        r2 = Range(0.5, frozenset({2, 3}))
        assert r1 < r2
        assert r2 > r1
        assert -r1 > -r2
        assert -r2 < -r1
