import numpy as np
import pytest
from prescribed_sip.experiments.targets import make_target,bounded_pps,seed_for
from prescribed_sip.core.objective import PairObjective


@pytest.mark.parametrize('kind',['arbitrary','sparse_support','diffuse','srs','conditional_poisson','stratified','dsd','infeasible_small'])
def test_independent_targets_have_fixed_size_row_sums(kind):
    target=make_target(8,3,kind,seed=123,geometry='1d')
    assert np.all(target.coords[:,1]==0)
    assert target.pi.sum()==pytest.approx(3.,abs=1e-12)
    assert np.all((target.pi>0)&(target.pi<=1))
    if kind.startswith('infeasible'):
        assert target.feasible_known is False
    else:
        matrix=target.sip
        np.testing.assert_allclose(matrix.sum(axis=1)-np.diag(matrix),2*target.pi,atol=1e-10)


def test_sparse_large_target_does_not_allocate_dense_sip():
    target=make_target(250,5,'random_equal',seed=1,geometry='1d',density=.01,pair_structure='nearby')
    assert isinstance(target.sip,dict)
    assert len(target.sip)==round(.01*250*249/2)
    assert target.pi.sum()==pytest.approx(5)


def test_bounded_pps_handles_certainty_units_and_seeds_are_stable():
    pi=bounded_pps(np.array([1000.,1.,1.,1.]),2)
    np.testing.assert_allclose(pi,[1.,1/3,1/3,1/3],atol=1e-12)
    assert seed_for(1,'a',2,'GFS')==seed_for(1,'a',2,'GFS')
    assert seed_for(1,'a',2,'GFS')!=seed_for(1,'a',3,'GFS')
