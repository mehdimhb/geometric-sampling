"""Independent moment checks catch sign errors and loss of design validity."""
import itertools
import numpy as np
import pytest
from prescribed_sip.core.design import Support, InvariantError, exchange_direction
from prescribed_sip.core.objective import PairObjective


def fixture():
    return Support(5, 2, {(0, 1): .3, (2, 3): .3, (0, 4): .2, (1, 4): .2})


def independent_moments(support):
    x = np.array([[int(i in s) for i in range(support.N)] for s in support.mass], float)
    p = np.array(list(support.mass.values()))
    return p @ x, x.T @ (p[:, None] * x)


def test_exchange_preserves_moments_and_exact_local_sign():
    d = fixture()
    pi, before = independent_moments(d)
    c = exchange_direction((0, 1), (2, 3), 0, 2)
    assert c == {(0, 1): -1, (0, 3): 1, (1, 2): 1, (2, 3): -1}
    d.exchange((0, 1), (2, 3), 0, 2, .1)
    after_pi, after = independent_moments(d)
    np.testing.assert_allclose(after_pi, pi, atol=1e-15)
    for i,j in itertools.combinations(range(5), 2):
        assert after[i,j]-before[i,j] == pytest.approx(.1*c.get((i,j),0), abs=1e-15)
    assert d.validate(pi)['max_FIP_error'] < 1e-15
    assert sum(d.mass.values()) == pytest.approx(1)


def test_coincident_destinations_and_full_step_are_aggregated():
    d = Support(3, 2, {(0,1): .5, (0,2): .5})
    d.exchange((0,1),(0,2),1,2,.5)
    assert d.mass == {(0,1): .5, (0,2): .5}
    assert exchange_direction((0,1),(0,2),1,2) == {}


@pytest.mark.parametrize('bad', [{(0,):1.}, {(0,1):-.1,(1,2):1.1}, {(0,1):.9}])
def test_invalid_design_is_rejected(bad):
    with pytest.raises(InvariantError):
        Support(3,2,bad).validate(np.array([1.,1.,0.]))


def test_reject_invalid_exchange_without_mutation():
    d=fixture(); snapshot=d.mass.copy()
    with pytest.raises(ValueError): d.exchange((0,1),(2,3),0,2,.31)
    assert d.mass == snapshot


def test_analytic_epsilon_beats_grid_and_increment_matches_independent_sip():
    d=fixture(); _, sip=independent_moments(d)
    target=sip.copy(); target[0,1]-=.12; target[1,0]-=.12
    obj=PairObjective(5,target); obj.initialize(d)
    direction=exchange_direction((0,1),(2,3),0,2)
    eps,delta=obj.step(direction,.3,'optimal',np.random.default_rng(1))
    assert eps == pytest.approx(.03)
    grid=np.linspace(0,.3,1001)
    losses=[]
    for e in grid:
        clone=d.copy(); clone.exchange((0,1),(2,3),0,2,float(e))
        _, cur=independent_moments(clone)
        losses.append(sum((cur[i,j]-target[i,j])**2 for i,j in itertools.combinations(range(5),2)))
    assert obj.D2+delta <= min(losses)+1e-14
    initial=obj.D2
    obj.apply(direction,eps); d.exchange((0,1),(2,3),0,2,eps)
    assert obj.D2 == pytest.approx(initial+delta,abs=1e-14)
    assert obj.recompute_error(d) < 1e-15


def test_sparse_objective_stores_only_selected_pairs_and_uses_targeted_denominator():
    d=fixture(); target={(0,1):.2,(0,3):.1}
    obj=PairObjective(5,target); obj.initialize(d)
    assert len(obj.current)==2
    assert obj.D2 == pytest.approx(.02)
    assert obj.metrics()['target_RMSE2'] == pytest.approx(.1)
    assert obj.metrics()['weighted_RMSE2'] == pytest.approx(np.sqrt(.02/10))
    assert np.isnan(obj.metrics()['RMSE2'])
    eps,delta=obj.step(exchange_direction((0,1),(2,3),0,2),.3,'optimal',np.random.default_rng(1))
    assert eps == pytest.approx(.1)
    assert delta == pytest.approx(-.02)
