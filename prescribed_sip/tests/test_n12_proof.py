"""Fixed-target proof tests: conservation, independent moments, and search precision."""
import numpy as np
import pytest
from graphical_sampling.search import GreedyBestFirstSearch
from prescribed_sip.experiments.n12_targets import independent_sip, build_case, make_target, PAIRS
from prescribed_sip.methods.n12_original import precision_run, run_original
from prescribed_sip.core.design import Support


def test_independent_pair_moments_hand_example():
    support=Support(4,2,{(0,1):.25,(2,3):.75})
    matrix=independent_sip(support)
    assert matrix[0,1]==.25 and matrix[2,3]==.75 and matrix[0,2]==0
    assert np.diag(matrix).tolist()==[.25,.25,.75,.75]


@pytest.mark.parametrize('case',['E','U'])
def test_target_walk_preserves_constraints_and_selects_changed_snapshot(case):
    pi,x,initial=build_case(case)
    target,trajectory,metadata=make_target(initial,pi,seed=191,exchanges=200,interval=50)
    target.validate(pi)
    sip=independent_sip(target)
    assert np.max(abs(sip.sum(axis=1)-np.diag(sip)-3*pi))<1e-10
    assert np.triu(sip,1).sum()==pytest.approx(6,abs=1e-10)
    assert metadata['d_rho']>=.8*max(row['d_rho'] for row in trajectory)-1e-14
    assert metadata['Hstar']>metadata['H0'] and metadata['initial_RMSE']>0
    assert np.min(pi)>.1 and np.max(pi)<.6
    if case=='U': assert np.allclose(pi,4*x/x.sum(),rtol=0,atol=1e-15)


def test_precision_sensitivity_reuses_same_bytecode_without_global_mutation():
    original=GreedyBestFirstSearch.run
    exact=precision_run(original)
    assert exact.__code__ is original.__code__
    assert exact.__globals__ is not original.__globals__
    # Catch identity binding accidentally returning nine-decimal quantization.
    assert exact.__globals__['round'](1e-12,9)==1e-12
    assert round(1e-12,9)==0
    assert GreedyBestFirstSearch.run is original


def test_original_run_budget_reproducible_trace_and_valid_final_support():
    pi,x,initial=build_case('U')
    target,_,_=make_target(initial,pi,seed=51,exchanges=50,interval=50)
    sip=independent_sip(target)
    a,sa=run_original(pi,sip,seed=17,evaluations=40,high_precision=False,record_interval=10)
    b,sb=run_original(pi,sip,seed=17,evaluations=40,high_precision=False,record_interval=10)
    assert a['candidate_evaluations']<=40
    assert a['final_D2']==pytest.approx(b['final_D2'],abs=1e-15)
    assert sa.mass==sb.mass
    assert a['trace'][0]['candidates']==0 and a['trace'][-1]['candidates']==a['candidate_evaluations']
    assert a['validation']['max_FIP_error']<1e-10
    pairs=np.triu_indices(12,1)
    assert a['final_D2']==pytest.approx(np.sum((independent_sip(sa)[pairs]-sip[pairs])**2),abs=1e-14)


def test_fitted_projection_dsd_returns_valid_enumerated_design():
    from prescribed_sip.n12_proof import fit_dsd
    pi,x,initial=build_case('U')
    target,_,_=make_target(initial,pi,seed=51,exchanges=50,interval=50)
    sip=independent_sip(target)
    support,row=fit_dsd(pi,sip,seed=15,evaluations=30)
    assert row['candidate_evaluations']==30
    validation=support.validate(pi)
    assert validation['max_FIP_error']<1e-10
    loss=np.mean((independent_sip(support)[PAIRS]-sip[PAIRS])**2)
    assert np.sqrt(loss)<=row['details']['initial_fixed_order_RMSE']+1e-12
    assert row['details']['validation']['max_SIP_recompute_error']<1e-12
