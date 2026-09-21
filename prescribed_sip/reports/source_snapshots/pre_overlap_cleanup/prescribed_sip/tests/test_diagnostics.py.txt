import itertools
import numpy as np
import pytest
from prescribed_sip.experiments.targets import make_target
from prescribed_sip.core.design import Support
from prescribed_sip.experiments.diagnostics import sparse_full_diagnostics


def test_sparse_full_diagnostics_match_independent_full_matrix_without_dilution():
    sparse=make_target(4,2,'srs',seed=1,geometry='1d',density=.25)
    d=Support(4,2,{(0,1):.5,(2,3):.5})
    result=sparse_full_diagnostics(sparse,d,d,method='GFS-Directed',block_size=1)
    # Two errors .5-1/6 and four errors 0-1/6: SSE=1/3, M=6.
    assert result['final_RMSE2']==pytest.approx(np.sqrt(1/18))
    assert result['final_Dinf']==pytest.approx(1/3)
    assert result['full_diagnostics_seconds']>=0


@pytest.mark.parametrize('kind',['random_equal','random_unequal','stratified','dsd'])
def test_streaming_diagnostics_same_target_as_generator(kind):
    sparse=make_target(25,5,kind,seed=19,geometry='1d',density=.01)
    full=make_target(25,5,kind,seed=19,geometry='1d')
    from prescribed_sip.methods.package_gfs import initial_design,seeded_package,extract_support
    with seeded_package(1): d=extract_support(initial_design(sparse.pi,sparse.coords))
    result=sparse_full_diagnostics(sparse,d,d,method='GFS-Directed',block_size=4)
    x=np.array([[int(i in s) for i in range(25)] for s in d.mass],float);p=np.array(list(d.mass.values()))
    sip=x.T@(p[:,None]*x);i,j=np.triu_indices(25,1)
    assert result['final_RMSE2']==pytest.approx(np.sqrt(np.mean((sip[i,j]-full.sip[i,j])**2)),abs=1e-12)
