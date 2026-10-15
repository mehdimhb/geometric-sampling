"""Preserve the historical score definition and provide opt-in seeded replay."""
import numpy as np
import pytest
from graphical_sampling import Population
from graphical_sampling.index import DensityDisparity


@pytest.mark.parametrize('representative', ['nmeans','nmedoids'])
def test_seeded_density_score_replays_across_workers_and_batches(representative):
    x = np.random.default_rng(14).random((24, 2))
    pop = Population(x, np.full(24, .25))
    samples = np.array([[0,3,6,10,15,23],[1,5,8,12,17,20],[0,3,6,10,15,23]])
    opts = dict(representative=representative, random_state=31)
    serial = DensityDisparity(pop,n_jobs=1,**opts)
    np.random.seed(103); before=np.random.get_state()
    values,densities=serial.score(samples,return_densities=True)
    parallel=DensityDisparity(pop,n_jobs=2,**opts).score(samples)
    np.testing.assert_array_equal(values,parallel)
    np.testing.assert_array_equal(values,serial.score(samples))
    np.testing.assert_array_equal(values[::-1],serial.score(samples[::-1]))
    assert values[0] == values[2] and np.isfinite(values).all()
    for (f,g),v in zip(densities,values):
        den=np.sqrt(2)*np.sqrt(f*f+g*g)
        dnet=np.clip((f-g)/den/np.sin(np.pi/8),-1,1).mean()
        eta=np.clip((1-(f+g)/den)/(1-np.cos(np.pi/8)),-1,1).mean()
        assert v == pytest.approx((1-eta)*dnet+eta*np.sign(dnet),abs=3e-15)
    after=np.random.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])


def test_existing_unseeded_call_and_return_densities_remain_available():
    x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    pop=Population(x,np.full(4,.5))
    scores,densities=DensityDisparity(pop,n_jobs=1).score(np.array([[0,3]]),True)
    assert scores.shape==(1,) and len(densities)==1
    assert densities[0][0].shape==(4,) and densities[0][1].shape==(4,)


@pytest.mark.parametrize('state', [-1, True, 1.2, 2**32])
def test_density_seed_validation(state):
    pop=Population(np.array([[0.,0.],[1.,1.]]),np.ones(2))
    with pytest.raises(ValueError,match='random_state'):
        DensityDisparity(pop,random_state=state)


def test_historical_process_override_keeps_its_signature():
    class HistoricalScorer(DensityDisparity):
        def _process_one_sample(self, sample_indices):
            return float(sample_indices[0]), self.original_density

    pop = Population(np.array([[0., 0.], [1., 1.]]), np.ones(2))
    scorer = HistoricalScorer(pop, 1, 1e-9, 100, 1e-4, 'nmedoids')
    np.testing.assert_array_equal(scorer.score(np.array([[0, 1], [1, 0]])), [0., 1.])


def test_medoid_exact_tie_uses_probability_then_population_index():
    pop = Population(np.array([[0., 0.], [1., 0.]]), np.array([.5, .5]))
    scorer = DensityDisparity(pop, n_jobs=1)
    medoids, indices = scorer._cluster_medoids(np.zeros(2, dtype=int), np.array([[.5, 0.]]))
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_array_equal(medoids, pop.coords[[0]])
