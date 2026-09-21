import numpy as np
import pytest
from prescribed_sip.experiments.targets import make_target,cp_moments
from prescribed_sip.methods.exact import space_for
from prescribed_sip.methods.package_gfs import initial_design,seeded_package,run_package,extract_support
from prescribed_sip.core.objective import PairObjective


def test_density_and_geometry_do_not_change_hidden_target_or_fip():
    full=make_target(25,5,'random_unequal',91,geometry='1d')
    sparse=make_target(25,5,'random_unequal',91,geometry='planar',density=.1)
    np.testing.assert_array_equal(full.pi,sparse.pi)
    for (i,j),v in sparse.sip.items(): assert full.sip[i,j]==v


def test_conditional_poisson_log_dp_matches_enumeration():
    odds=np.array([.1,.4,1.,2.,7.]); space=space_for(5,3)
    p=np.array([np.prod(odds[list(s)]) for s in space.samples]); p/=p.sum()
    pi,sip=cp_moments(odds,3,space.pair_list)
    np.testing.assert_allclose(pi,space.units@p,atol=1e-13)
    np.testing.assert_allclose(list(sip.values()),space.pairs@p,atol=1e-13)


def test_package_method_seed_controls_initial_heap_streams():
    pi=np.full(8,3/8); coords=np.column_stack([np.arange(8),np.zeros(8)])
    target=np.full((8,8),6/56); np.fill_diagonal(target,pi)
    with seeded_package(1): d=initial_design(pi,coords)
    first=run_package(d,PairObjective(8,target),pi,seed=99,evaluations=50,seconds=10)
    # Reusing the same design must not carry a heap RNG advanced by the first run.
    second=run_package(d,PairObjective(8,target),pi,seed=99,evaluations=50,seconds=10)
    assert first.support.mass==second.support.mass


def test_certainty_unit_survives_fractional_wrap_and_order_changes():
    # End/start remainders .1 and 1.1-1 differ by roundoff; a certainty unit
    # cannot be removed by their event ordering on any positive-mass interval.
    pi=np.array([.1,1.,.2,.3,.4]); coords=np.column_stack([np.arange(5),np.zeros(5)])
    with seeded_package(123):
        d=initial_design(pi,coords)
    support=extract_support(d)
    assert all(1 in s for s in support.mass)
    support.validate(pi)
    target=make_target(5,2,'srs',1).sip
    # Use a feasible target with the same FIP, from the known fixed certainty design.
    target=np.outer(pi,pi); np.fill_diagonal(target,pi)
    for i in range(5):
        for j in range(i+1,5): target[i,j]=target[j,i]=pi[i] if j==1 else pi[j] if i==1 else 0.
    outcome=run_package(d,PairObjective(5,target),pi,seed=51,evaluations=30,seconds=2)
    assert outcome.validation['max_FIP_error']<1e-10
    assert all(1 in s for s in outcome.support.mass)
