import numpy as np
import pytest
from prescribed_sip.core.design import Support
from prescribed_sip.core.objective import PairObjective
from prescribed_sip.methods.exchange_search import run_exchange,random_candidate


def problem():
    d=Support(4,2,{(0,1):.5,(2,3):.5})
    target=np.full((4,4),1/6); np.fill_diagonal(target,.5)
    return d,target


@pytest.mark.parametrize('method',['GFS-Random','GFS-Greedy','GFS-Directed','GFS-Advanced'])
def test_budget_and_invariants(method):
    d,target=problem()
    result=run_exchange(d,PairObjective(4,target),d.fip(),seed=10,evaluations=101,seconds=3,method=method,batch=10)
    assert result.candidate_moves<=101
    assert result.validation['max_FIP_error']<1e-10
    assert result.validation['max_SIP_incremental_error']<1e-10
    assert result.final['RMSE2']<=result.initial['RMSE2'] if method!='GFS-Random' else True
    if method=='GFS-Advanced': assert result.details['starts_completed']>=2


def test_directed_recovery_known_small_design():
    d,target=problem()
    result=run_exchange(d,PairObjective(4,target),d.fip(),seed=1,evaluations=1000,seconds=5,method='GFS-Directed',batch=10)
    assert result.final['RMSE2']<1e-8


def test_epsilon_ablation_random_policy_has_valid_nonoptimal_trajectory():
    d,target=problem()
    result=run_exchange(d,PairObjective(4,target),d.fip(),seed=8,evaluations=100,seconds=3,method='GFS-Greedy',epsilon='random',batch=10)
    assert result.accepted_moves>0
    assert result.final['D2']<result.initial['D2']
