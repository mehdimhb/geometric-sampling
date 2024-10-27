import unittest
from fast.structs import Range


class RangeTestCase(unittest.TestCase):
    def test_almost_zero(self):
        r = Range(1e-10, frozenset({1, 2, 3}))
        assert r.almost_zero()

    def test_eq(self):
        r1 = Range(0.3, frozenset({1, 2, 3}))
        r2 = Range(0.5, frozenset({1, 2, 3}))
        assert r1 == r2

        r3 = Range(0.3, frozenset({1, 2}))
        assert r1 != r3

    def test_compare(self):
        r1 = Range(0.3, frozenset({1, 2}))
        r2 = Range(0.5, frozenset({2, 3}))
        assert r1 < r2
        assert r1 <= r2
        assert r2 > r1
        assert r2 >= r1
        assert -r1 > -r2
        assert -r2 < -r1
        assert -r1 >= -r2
        assert -r2 <= -r1

    def test_add(self):
        r1 = Range(0.3, frozenset({1, 2, 3}))
        r2 = Range(0.5, frozenset({1, 2, 3}))
        r3 = r1 + r2
        assert r3.length == 0.8
        assert r3.ids == frozenset({1, 2, 3})
