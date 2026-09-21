"""Regression checks for probability-aware GBFS duplicate detection."""
from types import SimpleNamespace

from graphical_sampling.search import GreedyBestFirstSearch
from graphical_sampling.structs import _MaxHeap, _Sample


def design_with_fragments(fragments):
    heap = _MaxHeap()
    for ids, mass in fragments:
        heap.push(_Sample(mass, frozenset(ids)))
    return SimpleNamespace(_heaps=[heap])


def test_same_support_different_probabilities_are_distinct():
    a = design_with_fragments([({0, 1}, .4), ({2, 3}, .4),
                               ({0, 2}, .1), ({1, 3}, .1)])
    b = design_with_fragments([({0, 1}, .3), ({2, 3}, .3),
                               ({0, 2}, .2), ({1, 3}, .2)])
    assert GreedyBestFirstSearch._design_key(a) != GreedyBestFirstSearch._design_key(b)


def test_split_and_merged_fragments_share_key():
    a = design_with_fragments([({0, 1}, .125), ({0, 1}, .375), ({2, 3}, .5)])
    b = design_with_fragments([({2, 3}, .5), ({0, 1}, .5)])
    assert GreedyBestFirstSearch._design_key(a) == GreedyBestFirstSearch._design_key(b)


def test_snapshot_remains_safe_after_heap_changes():
    a = design_with_fragments([({0, 1}, .5), ({2, 3}, .5)])
    key = GreedyBestFirstSearch._design_key(a)
    closed = {key}
    a._heaps[0].push(_Sample(.125, frozenset({0, 2})))
    assert key in closed
    assert GreedyBestFirstSearch._design_key(a) not in closed
