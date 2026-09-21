import inspect
import random
import numpy as np
from graphical_sampling.search import GreedyBestFirstSearch
from prescribed_sip.methods.package_gfs import Float64Design, seeded_package, initial_design, run_package
from prescribed_sip.core.objective import PairObjective


def test_raw_export_retains_double_precision_and_no_normalization():
    from graphical_sampling.structs import _Sample,_MaxHeap
    with seeded_package(1): d=initial_design(np.array([.5,.5,.5,.5]),np.column_stack([np.arange(4),np.zeros(4)]))
    d._heaps=[_MaxHeap()]
    d._heaps[0].push(_Sample(.123456789012345,frozenset({0,1})))
    d._heaps[0].push(_Sample(.876543210987654,frozenset({2,3})))
    Float64Design._reset_stats(d)
    samples,p=d.all_samples_and_probs
    assert p.dtype==np.float64
    assert p[0] == .123456789012345 or p[1] == .123456789012345
    assert p.sum() < 1.


def test_seeded_package_restores_rng_and_main_search_is_inherited():
    before=random.getstate(); default=np.random.default_rng
    with seeded_package(91): a=np.random.default_rng().random(5); x=random.random()
    with seeded_package(91): b=np.random.default_rng().random(5); y=random.random()
    np.testing.assert_array_equal(a,b); assert x==y
    assert random.getstate()==before
    assert np.random.default_rng is default


def test_package_optimizer_reproducible_and_line_coordinates_preserved():
    pi=np.full(8,3/8); coords=np.column_stack([np.arange(8),np.zeros(8)])
    target=np.full((8,8),3*2/(8*7)); np.fill_diagonal(target,pi)
    result=[]
    for _ in range(2):
        with seeded_package(9): d=initial_design(pi,coords)
        assert isinstance(d,Float64Design)
        assert np.all(d.pop.coords[:,1]==0)
        result.append(run_package(d,PairObjective(8,target),pi,seed=51,evaluations=80,seconds=5,batch=10))
    assert result[0].support.mass==result[1].support.mass
    assert result[0].candidate_moves<=80
    assert result[0].validation['max_FIP_error']<1e-10
    assert result[0].details['search_implementation']=='graphical_sampling.search.GreedyBestFirstSearch.run'
