"""Output modes must not impose spatial diagnostic costs on SIP searches."""
from types import SimpleNamespace

import numpy as np
import pytest

from graphical_sampling import Design, Population
from graphical_sampling.criteria import SIPMatchingCriteria
from graphical_sampling.search import GreedyBestFirstSearch
from graphical_sampling.structs import _MaxHeap, _Sample


def search_pair():
    pop = Population(np.column_stack([np.arange(4), np.zeros(4)]),
                     np.full(4, .5), variable=np.ones(4), n=2)
    designs = []
    for large, small in [(.4, .1), (.3, .2)]:
        design = Design(pop)
        design.order.clusters = [SimpleNamespace(zones=[None])]
        heap = _MaxHeap()
        for ids, mass in [({0, 1}, large), ({2, 3}, large),
                          ({0, 2}, small), ({1, 3}, small)]:
            heap.push(_Sample(mass, frozenset(ids)))
        design._heaps = [heap]
        designs.append(design)
    a, b = designs
    return GreedyBestFirstSearch([a], SIPMatchingCriteria(b.sip)), b


def run_search(search, mode, details=False):
    search.run(max_iterations=1, max_open_set_size=10, top_k=1,
               num_new_order_nodes=0, num_new_exchange_nodes=1,
               num_clusters_range=(1, 1), num_zones_range=(1, 1),
               num_changes_range=(1, 1), num_zone_changes=0, n_jobs=1,
               output_mode=mode, objective_name='SIP RMSE', output_details=details)


@pytest.mark.parametrize('mode,details', [('objective', False), ('objective', True),
                                         ('silent', False), ('silent', True)])
def test_nonspatial_output_skips_diagnostics(monkeypatch, capsys, mode, details):
    search, target = search_pair()
    monkeypatch.setattr(search, '_exchange_neighbors', lambda *args: iter([('exchange', target)]))

    def forbidden(design):
        raise AssertionError('SIP output must not calculate spatial diagnostics')

    for name in ['moran', 'nht_variance', 'entropy']:
        monkeypatch.setattr(Design, name, property(forbidden))
    run_search(search, mode, details)
    output = capsys.readouterr().out
    assert search.best_criteria_value == 0
    if mode == 'silent':
        assert output == ''
    else:
        assert 'SIP RMSE 0.00000000' in output and 'BEST 0.00000000' in output
        assert 'MI ' not in output and 'SIZ ' not in output and 'VAR_' not in output


def test_spatial_output_retains_diagnostics(monkeypatch, capsys):
    search, target = search_pair()
    monkeypatch.setattr(search, '_exchange_neighbors', lambda *args: iter([('exchange', target)]))
    monkeypatch.setattr(Design, 'moran', property(lambda design: (.25, 0)))
    run_search(search, 'spatial')
    output = capsys.readouterr().out
    assert 'MI 0.25000' in output and 'SIZ ' in output and '1C x 1Z' in ' '.join(output.split())
    assert 'VAR_aux' in output
