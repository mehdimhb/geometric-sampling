from itertools import combinations
import numpy as np
import pytest
from prescribed_sip.core.objective import PairObjective
from prescribed_sip.methods.exact import EnumeratedSpace, exact_benchmark
from prescribed_sip.methods.cp2 import fit_cp2
from prescribed_sip.methods.sinha import sinha_parameters, sinha_moments, sinha_support
from prescribed_sip.methods.dsd import projection_factor, projection_moments


def test_lp_recovers_srs_and_infeasible_qp_has_certificate():
    pi=np.full(5,.4); target=np.full((5,5),.1); np.fill_diagonal(target,pi)
    obj=PairObjective(5,target)
    result=exact_benchmark(5,2,pi,obj,seconds=5)
    assert result['feasible'] is True
    assert result['optimal_D2']<1e-16
    assert result['primal_error']<1e-10
    # n=2: p_{01} must be nonnegative. Negative target cannot be feasible.
    target[0,1]=target[1,0]=-.05
    result=exact_benchmark(5,2,pi,PairObjective(5,target),seconds=5)
    assert result['feasible'] is False
    assert result['optimal_D2']>=.0025-1e-10
    assert result['certificate_gap']<1e-7
    assert result['primal_error']<1e-10


def test_cp2_recovers_known_exponential_example_and_preserves_fip():
    space=EnumeratedSpace(5,2)
    theta=np.linspace(-.7,.7,10)
    p=np.exp(space.pairs.T@theta); p/=p.sum()
    pi=space.units@p
    target=np.zeros((5,5)); i,j=np.triu_indices(5,1)
    target[i,j]=target[j,i]=space.pairs@p; np.fill_diagonal(target,pi)
    outcome=fit_cp2(5,2,pi,PairObjective(5,target),evaluations=2000,seconds=5)
    assert outcome.final['RMSE2']<1e-7
    assert outcome.validation['max_FIP_error']<1e-10


def test_sinha_documented_pair_removal_example():
    # Always remove {0,1}, then take 2 of remaining {2,3,4} uniformly.
    b={(i,j):float((i,j)==(0,1)) for i,j in combinations(range(5),2)}
    pi,sip=sinha_moments(5,2,b)
    np.testing.assert_allclose(pi,[0,0,2/3,2/3,2/3],atol=1e-14)
    assert sip[(2,3)] == pytest.approx(1/3)
    recovered=sinha_parameters(5,2,pi,sip)
    for pair in b: assert recovered[pair]==pytest.approx(b[pair],abs=1e-14)
    support=sinha_support(5,2,recovered)
    assert support.mass==pytest.approx({(2,3):1/3,(2,4):1/3,(3,4):1/3})


def test_sinha_rejects_negative_pair_removal_mixture():
    with pytest.raises(ValueError): sinha_support(5,2,{(0,1):-1.,(2,3):2.})


def test_projection_dsd_known_diagonal_and_enumerated_moments():
    pi=np.array([.5,.25,.75,.6,.9]) # sum 3
    V=projection_factor(pi)
    np.testing.assert_allclose(V.T@V,np.eye(3),atol=1e-12)
    K=V@V.T
    np.testing.assert_allclose(np.diag(K),pi,atol=1e-12)
    pairs=list(combinations(range(5),2))
    moments=projection_moments(V,pairs)
    p={s:float(np.linalg.det(K[np.ix_(s,s)])) for s in combinations(range(5),3)}
    assert sum(p.values())==pytest.approx(1.,abs=1e-12)
    for i,j in pairs:
        assert moments[(i,j)]==pytest.approx(sum(q for s,q in p.items() if i in s and j in s),abs=1e-12)


def test_lp_dual_bound_corrects_infeasible_reduced_costs():
    from scipy import sparse
    from prescribed_sip.methods.exact import conservative_linear_dual_bound
    A=sparse.csc_matrix([[1.,1.]])
    # Candidate dual lambda=3 violates g-A'lambda >=0 for g=[1,2].
    bound,diagnostics=conservative_linear_dual_bound(np.array([1.,2.]),A,np.array([1.]),np.array([3.]))
    assert bound<=1. # exact linear minimum over the simplex
    assert bound>=1.-1e-12
    assert diagnostics['oracle_dual_violation_before_correction']==pytest.approx(2.)
