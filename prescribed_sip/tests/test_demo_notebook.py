"""Execute the educational notebook's real cells without launching large searches."""
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings(
    'ignore:FigureCanvasAgg is non-interactive:UserWarning'
)

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/GFS_prescribed_SIP_demo.ipynb'


def execute_cells(overrides=None, through=None, notebook_path=NOTEBOOK):
    assert notebook_path.exists(), 'The requested executable teaching notebook is missing'
    notebook = json.loads(notebook_path.read_text())
    namespace = {'__name__': '__notebook_test__'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), namespace)
        tag = cell['metadata'].get('tags', [''])[0]
        if tag == 'settings':
            namespace.update({"N_REPETITIONS": 1, "DIRECTED_N_JOBS": 1, "DIRECTED_SAVE_RESULTS": False})
            namespace.update(overrides or {})
        if tag == through:
            break
    plt.close('all')
    return namespace


@pytest.mark.parametrize('fip_type', ['equal', 'unequal'])
def test_simple_target_cell_matches_long_walk_and_feeds_later_cells(fip_type):
    simple_path = NOTEBOOK.parents[2] / 'simulations/jupyters/GFS_prescribed_SIP_demo_simple.ipynb'
    settings = {'POPULATION': 'synthetic', 'N': 12, 'n': 4, 'FIP_TYPE': fip_type,
                'TARGET_EXCHANGES': 20, 'GFS_MAX_EVALUATIONS': 60,
                'RUN_GFS_DIRECTED': True, 'DIRECTED_MAX_EVALUATIONS': 100,
                'RUN_CP2': False, 'RUN_DSD': False, 'RUN_EXACT_QP': False,
                'GFS_AUTO_RESTART': False}
    original = execute_cells(settings, through='target')
    simple = execute_cells(settings, notebook_path=simple_path)
    np.testing.assert_array_equal(simple['Pi_star'], original['Pi_star'])
    assert simple['distribution'](simple['p_star']) == original['distribution'](original['p_star'])
    assert simple['accepted_exchanges'] == 20
    assert simple['attempted_exchanges'] == original['attempted_exchanges']
    assert simple['target_trace'][-1]['exchanges'] == 20
    assert simple['target_history'].iloc[-1]['exchanges'] == 20
    assert simple['current_design'] is simple['p_star']
    assert simple['target_runtime'] > 0
    assert simple['summary']['Method'].tolist() == ['Initial', 'Original GFS', 'GFS-Directed']


def test_independent_moments_and_capped_inclusion_probabilities():
    demo = execute_cells(through='helpers')
    distribution = SimpleNamespace(N=4, n=2, mass={(0, 1): .25, (2, 3): .75})
    expected = np.array([[.25, .25, 0, 0], [.25, .25, 0, 0],
                         [0, 0, .75, .75], [0, 0, .75, .75]])
    np.testing.assert_allclose(demo['compute_SIP'](distribution), expected, atol=1e-15)
    np.testing.assert_allclose(demo['inclusion_probabilities'](np.array([100., 1., 1., 1.]), 2),
                               [1., 1/3, 1/3, 1/3], atol=1e-12)
    demo['validate_design'](distribution, np.array([.25, .25, .75, .75]))


def test_small_demo_preserves_design_and_obeys_candidate_budget():
    settings = {'TARGET_EXCHANGES': 20, 'GFS_MAX_EVALUATIONS': 120,
                'RUN_CP2': False, 'RUN_DSD': False, 'RUN_EXACT_QP': False,
                'RUN_GFS_DIRECTED': False, 'GFS_AUTO_RESTART': False}
    first = execute_cells(settings)
    second = execute_cells(settings)
    assert first['accepted_exchanges'] == 20
    assert 0 < first['gfs_result']['evaluations'] <= 120
    assert first['summary']['Method'].tolist() == ['Initial', 'Original GFS']
    np.testing.assert_allclose(first['Pi_star'], second['Pi_star'], atol=0, rtol=0)
    np.testing.assert_allclose(first['gfs_result']['Pi'], second['gfs_result']['Pi'], atol=0, rtol=0)
    assert first['validate_design'](first['gfs_result']['design'], first['pi'])['FIP max error'] < 1e-10


def test_directed_comparison_uses_same_target_obeys_budget_and_preserves_fip():
    demo = execute_cells({'TARGET_EXCHANGES': 20, 'GFS_MAX_EVALUATIONS': 60,
                          'RUN_GFS_DIRECTED': True, 'DIRECTED_MAX_EVALUATIONS': 200,
                          'RUN_CP2': False, 'RUN_DSD': False, 'RUN_EXACT_QP': False,
                          'GFS_AUTO_RESTART': False})
    assert 'GFS-Directed' in demo['summary']['Method'].tolist()
    result = demo['directed_result']
    assert 0 < result.candidate_moves <= 200
    independent_Pi = demo['compute_SIP'](result.support)
    difference = (independent_Pi - demo['Pi_star'])[np.triu_indices(12, 1)]
    expected_RMSE = np.sqrt(np.sum(difference ** 2) / 66)
    assert abs(expected_RMSE - result.final['target_RMSE2']) < 1e-12
    assert expected_RMSE <= demo['baseline_rmse'] + 1e-12
    assert demo['validate_design'](result.support, demo['pi'])['FIP max error'] < 1e-10
    assert abs(demo['directed_convergence'].iloc[-1]['RMSE'] - expected_RMSE) < 1e-12


def test_synthetic_284_builds_without_enumerating_samples():
    demo = execute_cells({'N': 284, 'n': 40, 'FIP_TYPE': 'unequal'}, through='initial')
    assert len(demo['pi']) == 284
    assert abs(sum(demo['pi']) - 40) < 1e-10
    assert demo['Pi0'].shape == (284, 284)
    assert not demo['enumeration_allowed'](284, 40)


def test_mu284_reports_actual_rows_and_preserves_certainty_units():
    demo = execute_cells({'POPULATION': 'MU284', 'N': 284, 'n': 40,
                          'FIP_TYPE': 'unequal'}, through='initial')
    assert demo['N'] == 281  # Real repository data; never fabricate missing units.
    np.testing.assert_array_equal(demo['x'], demo['population_df']['P75'].to_numpy(dtype=float))
    assert demo['validate_design'](demo['p0'], demo['pi'])['FIP max error'] < 1e-10


def test_large_search_requires_opt_in_above_thirty_minutes():
    demo = execute_cells({'TARGET_EXCHANGES': 5}, through='search_helpers')
    assert not demo['runtime_permitted'](1801., False)
    assert demo['runtime_permitted'](1801., True)
    assert demo['runtime_permitted'](60., False)


def test_parallel_scoring_cannot_silently_bypass_tracking_and_budget():
    demo = execute_cells({'TARGET_EXCHANGES': 5}, through='search_helpers')
    demo['search_parameters']['n_jobs'] = 2
    with pytest.raises(ValueError, match='n_jobs=1'):
        demo['run_gfs'](3, verbose=False)


def test_stagnation_restarts_continue_to_shared_budget_and_keep_best_result():
    demo = execute_cells({'TARGET_EXCHANGES': 20, 'GFS_STAGNATION_EVALUATIONS': 1,
                          'GFS_AUTO_RESTART': True, 'GFS_RESTART_EXCHANGES': 3,
                          'GFS_MAX_EMPTY_RESTARTS': 3}, through='search_helpers')
    first = demo['run_gfs'](120, verbose=False)
    assert first['evaluations'] == 120
    assert first['restarts'] > 0
    assert first['status'] == 'candidate budget reached'
    assert first['method'] == 'Original GFS + restarts'
    assert np.all(np.diff(first['trace']['RMSE']) <= 1e-12)
    assert demo['validate_design'](first['design'], demo['pi'])['FIP max error'] < 1e-10
    second = demo['run_gfs'](120, verbose=False)
    np.testing.assert_array_equal(first['Pi'], second['Pi'])


def test_real_frontier_exhaustion_restarts_and_restart_seeds_count_toward_budget():
    demo = execute_cells({'TARGET_EXCHANGES': 20, 'GFS_STAGNATION_EVALUATIONS': None,
                          'GFS_AUTO_RESTART': True, 'GFS_RESTART_EXCHANGES': 3,
                          'GFS_MAX_EMPTY_RESTARTS': 3}, through='search_helpers')
    demo['search_parameters'].update(max_open_set_size=1, stagnation_limit=1,
                                      num_new_order_nodes=0, num_new_exchange_nodes=1)
    result = demo['run_gfs'](120, verbose=False)
    assert result['evaluations'] == 120
    assert result['restarts'] > 0
    assert 'frontier exhausted' in result['restart_history']['reason'].tolist()
    assert result['trace'].iloc[-1]['evaluations'] == 120


def test_restart_disabled_retains_original_stop_and_empty_frontier_cannot_loop_forever():
    demo = execute_cells({'TARGET_EXCHANGES': 20, 'GFS_STAGNATION_EVALUATIONS': 1,
                          'GFS_AUTO_RESTART': False, 'GFS_RESTART_EXCHANGES': 0,
                          'GFS_MAX_EMPTY_RESTARTS': 3}, through='search_helpers')
    original = demo['run_gfs'](120, verbose=False)
    assert original['evaluations'] < 120
    assert original['restarts'] == 0
    assert original['method'] == 'Original GFS'
    demo['GFS_AUTO_RESTART'] = True
    demo['search_parameters'].update(num_new_order_nodes=0, num_new_exchange_nodes=0)
    empty = demo['run_gfs'](120, verbose=False)
    assert empty['evaluations'] == 0
    assert empty['restarts'] <= 3
    assert 'no candidates' in empty['status']
    timed = demo['run_gfs'](120, verbose=False, max_seconds=0.)
    assert timed['evaluations'] == 0
    assert 'time limit' in timed['status']


def test_directed_repetitions_report_independent_runs_and_mean_sample_sd():
    settings = {'TARGET_EXCHANGES': 15, 'GFS_MAX_EVALUATIONS': 20,
                'RUN_GFS_DIRECTED': True, 'N_REPETITIONS': 3,
                'DIRECTED_MAX_EVALUATIONS': 100, 'DIRECTED_SAVE_RESULTS': False,
                'RUN_CP2': False, 'RUN_DSD': False, 'RUN_EXACT_QP': False,
                'GFS_AUTO_RESTART': False}
    demo = execute_cells(settings)
    assert 'directed_runs' in demo, 'Notebook must retain all independent final results'
    runs = demo['directed_runs']
    assert len(runs) == 3 and runs['seed'].nunique() == 3
    i, j = np.triu_indices(12, 1)
    pi = demo['pi']
    denominator = np.sqrt(pi[i]*(1-pi[i])*pi[j]*(1-pi[j]))
    independent_rho = []
    independent_rmse = []
    for result in demo['directed_results']:
        diff = (demo['compute_SIP'](result.support) - demo['Pi_star'])[np.triu_indices(12, 1)]
        independent_rmse.append(np.sqrt(np.mean(diff ** 2)))
        independent_rho.append(np.sqrt(np.mean((diff / denominator) ** 2)))
        assert 0 < result.candidate_moves <= 100
        assert demo['validate_design'](result.support, demo['pi'])['FIP max error'] < 1e-10
    np.testing.assert_allclose(runs['RMSE'], independent_rmse, atol=1e-12)
    np.testing.assert_allclose(runs['RMSE_rho'], independent_rho)
    np.testing.assert_allclose(runs['Recovery_rho'], 100 * (1 - np.array(independent_rho) / demo['baseline_rho_error']))
    row = demo['summary'].set_index('Method').loc['GFS-Directed']
    assert row['Repetitions'] == 3
    assert row['RMSE'] == pytest.approx(np.mean(independent_rmse))
    assert row['RMSE SD'] == pytest.approx(np.std(independent_rmse, ddof=1))
    assert row['RMSE_rho SD'] == pytest.approx(np.std(independent_rho, ddof=1))
    assert {'RMSE_rho', 'Recovery_rho'}.issubset(demo['gfs_result']['trace'])
    assert demo['directed_convergence']['repetition'].nunique() == 3
    assert {'RMSE_rho', 'Recovery_rho'}.issubset(demo['directed_convergence'])
    repeated = execute_cells(settings)
    np.testing.assert_array_equal(runs['seed'], repeated['directed_runs']['seed'])
    np.testing.assert_array_equal(runs['RMSE'], repeated['directed_runs']['RMSE'])


def test_zero_pair_target_reports_undefined_normalization():
    demo = execute_cells({'N': 4, 'n': 1, 'TARGET_EXCHANGES': 0,
                          'GFS_MAX_EVALUATIONS': 10, 'RUN_GFS_DIRECTED': False,
                          'RUN_CP2': False, 'RUN_DSD': False, 'RUN_EXACT_QP': False})
    assert 'RMSE_rho' in demo['summary'], 'Dependence comparison columns are required'
    assert demo['baseline_rho_error'] == 0
    assert (demo['summary']['RMSE_rho'] == 0).all()
    assert demo['summary']['Recovery_rho'].isna().all()


def test_parallel_directed_repetitions_match_serial_seeds_and_results():
    settings = {'TARGET_EXCHANGES': 15, 'GFS_MAX_EVALUATIONS': 20,
                'RUN_GFS_DIRECTED': True, 'N_REPETITIONS': 3,
                'DIRECTED_MAX_EVALUATIONS': 100, 'DIRECTED_SAVE_RESULTS': False,
                'RUN_CP2': False, 'RUN_DSD': False, 'RUN_EXACT_QP': False,
                'GFS_AUTO_RESTART': False, 'DIRECTED_N_JOBS': 2}
    parallel = execute_cells(settings)
    assert parallel.get('directed_workers') == 2, 'Independent runs must use two processes'
    import os
    assert (parallel['directed_runs']['Worker PID'] != os.getpid()).all()
    serial = execute_cells({**settings, 'DIRECTED_N_JOBS': 1})
    columns = ['repetition', 'seed', 'RMSE', 'RMSE_rho', 'Recovery_rho', 'Evaluations', 'Status', 'Stop reason']
    pd = parallel['pd']
    pd.testing.assert_frame_equal(parallel['directed_runs'][columns], serial['directed_runs'][columns])
    assert parallel['directed_result'] is parallel['directed_results'][0]
    assert parallel['directed_convergence']['repetition'].nunique() == 3


@pytest.mark.parametrize('fip_type', ['equal', 'unequal'])
def test_target_permutation_preserves_unit_ids_and_restores_original_matrix_order(fip_type):
    demo = execute_cells({'TARGET_EXCHANGES': 0, 'FIP_TYPE': fip_type}, through='target')
    assert 'target_permutation' in demo, 'Target must start from a reproducibly permuted order'
    permutation = np.random.default_rng(demo['SEED']).permutation(demo['N'])
    np.testing.assert_array_equal(demo['target_permutation'], permutation)
    np.testing.assert_array_equal(demo['p0'].order.get()[:, 0].astype(int), np.arange(12))
    np.testing.assert_array_equal(demo['target_initial'].order.get()[:, 0].astype(int), permutation)
    # The package order stores ORIGINAL IDs; prescribed pi stays indexed by those IDs.
    np.testing.assert_array_equal(demo['target_initial'].pop.inclusions, demo['pi'])
    # Independent builder with physically reordered records gives the same ordered SIP.
    Population, Order = demo['Population'], demo['Order']
    pop = Population(coords=demo['coords'][permutation], inclusions=demo['pi'][permutation],
                     variable=demo['x'][permutation], n=None)
    pop._n = 4
    order = Order(pop)
    order.clusters = [demo['Cluster'](label=0, zones=[demo['Zone'](
        _shares=np.ones(12), _indices=np.arange(12), sort=list(range(12)))])]
    order.num_zones = order.num_splits = 1
    order._build_order(1)
    with demo['seeded_package'](demo['SEED']):
        expected = demo['Float64Design'].from_order(pop, order)
    expected_permuted = demo['compute_SIP'](expected)
    inverse = np.argsort(permutation)
    np.testing.assert_allclose(demo['Pi_star_permuted'], expected_permuted, atol=1e-12)
    np.testing.assert_allclose(demo['Pi_star'], expected_permuted[np.ix_(inverse, inverse)], atol=1e-12)
    np.testing.assert_allclose(np.diag(demo['Pi_star']), demo['pi'], atol=1e-12)
    assert all(len(s) == 4 for s in demo['distribution'](demo['p_star']))
    assert demo['p_star'] is not demo['p0']
    assert not np.allclose(demo['Pi_star'], demo['Pi0'])


def test_permuted_target_walk_reconstruction_keeps_original_start_and_is_reproducible():
    settings = {'TARGET_EXCHANGES': 20, 'FIP_TYPE': 'unequal',
                'GFS_MAX_EVALUATIONS': 40, 'DIRECTED_MAX_EVALUATIONS': 100,
                'N_REPETITIONS': 2, 'RUN_CP2': False, 'RUN_DSD': False,
                'RUN_EXACT_QP': False, 'GFS_AUTO_RESTART': False}
    first = execute_cells(settings)
    assert 'target_initial' in first, 'Permuted target design must be separate from reconstruction p0'
    second = execute_cells(settings)
    np.testing.assert_array_equal(first['target_permutation'], second['target_permutation'])
    np.testing.assert_array_equal(first['Pi_star'], second['Pi_star'])
    # Both optimizer traces must start at the original-order initial error.
    expected = np.sqrt(np.mean((first['Pi0'] - first['Pi_star'])[np.triu_indices(12, 1)] ** 2))
    assert first['gfs_result']['trace'].iloc[0]['RMSE'] == pytest.approx(expected)
    for _, history in first['directed_convergence'].groupby('repetition'):
        assert history.iloc[0]['RMSE'] == pytest.approx(expected)
    assert first['accepted_exchanges'] == 20
    np.testing.assert_allclose(first['Pi_star'].sum(axis=1), 4 * first['pi'], atol=1e-12)


def test_permuted_target_preserves_certainty_unit_ids_in_real_population():
    demo = execute_cells({'POPULATION': 'MU284', 'N': 284, 'n': 60,
                          'FIP_TYPE': 'unequal', 'TARGET_EXCHANGES': 0}, through='target')
    assert np.any(demo['pi'] == 1)
    np.testing.assert_allclose(np.diag(demo['Pi_star']), demo['pi'], atol=1e-12)
    certain_ids = set(np.flatnonzero(demo['pi'] == 1))
    assert all(certain_ids.issubset(s) and len(s) == 60
               for s, p in demo['distribution'](demo['p_star']).items() if p > 0)
    np.testing.assert_array_equal(demo['Pi_star'], demo['compute_SIP'](demo['p_star']))


def test_dependence_metrics_use_pairwise_standardization_and_complete_traces():
    demo = execute_cells({'TARGET_EXCHANGES': 15, 'FIP_TYPE': 'unequal',
                          'GFS_MAX_EVALUATIONS': 40, 'DIRECTED_MAX_EVALUATIONS': 150,
                          'N_REPETITIONS': 2, 'RUN_CP2': False, 'RUN_DSD': False,
                          'RUN_EXACT_QP': False})
    required = {'RMSE', 'RMSE_rho', 'Recovery_rho', 'Max SIP error',
                'Support size', 'Entropy', 'Rho pairs excluded'}
    assert required.issubset(demo['summary']), 'Dependence metrics must replace NRMSE in main results'
    assert not {'NRMSE', 'Recovery_N'}.intersection(demo['summary'])
    i, j = np.triu_indices(12, 1)
    pi = demo['pi']
    denominator = np.sqrt(pi[i]*(1-pi[i])*pi[j]*(1-pi[j]))
    target_rho = (demo['Pi_star'][i,j] - pi[i]*pi[j]) / denominator
    initial_rho = (demo['Pi0'][i,j] - pi[i]*pi[j]) / denominator
    baseline = np.sqrt(np.mean((initial_rho-target_rho)**2))
    errors = []
    for result in demo['directed_results']:
        Pi = demo['compute_SIP'](result.support)
        rho = (Pi[i,j]-pi[i]*pi[j]) / denominator
        error = np.sqrt(np.mean((rho-target_rho)**2))
        errors.append(error)
        assert result.final['RMSE_rho'] == pytest.approx(error)
    np.testing.assert_allclose(demo['directed_runs']['RMSE_rho'], errors)
    np.testing.assert_allclose(demo['directed_runs']['Recovery_rho'], 100*(1-np.array(errors)/baseline))
    row = demo['summary'].set_index('Method').loc['GFS-Directed']
    assert row['RMSE_rho'] == pytest.approx(np.mean(errors))
    assert row['RMSE_rho SD'] == pytest.approx(np.std(errors, ddof=1))
    for trace in [demo['gfs_result']['trace'], demo['directed_convergence']]:
        assert required.issubset(trace)
        assert {'runtime', 'evaluations'}.issubset(trace)
        assert trace.iloc[0]['RMSE_rho'] == pytest.approx(baseline)


def test_dependence_boundary_pairs_and_no_valid_pairs_are_explicit():
    from prescribed_sip.core.dependence import DependenceError
    pi = np.array([0., 1., 1e-14, 1.-1e-14, .2, .5, .8])
    metric = DependenceError(pi, boundary_tol=1e-12)
    residual = np.full(21, .03)
    expected = np.sqrt(np.mean([(.03/np.sqrt(.2*.8*.5*.5))**2,
                                (.03/np.sqrt(.2*.8*.8*.2))**2,
                                (.03/np.sqrt(.5*.5*.8*.2))**2]))
    assert metric.included == 3 and metric.excluded == 18
    assert metric.error(residual) == pytest.approx(expected)
    assert metric.recovery(expected/2, expected) == pytest.approx(50.)
    assert np.isnan(metric.recovery(0., 0.))
    empty = DependenceError(np.array([0., 1., 1e-14]))
    assert empty.included == 0 and empty.excluded == 3
    assert np.isnan(empty.error(np.zeros(3)))


def test_directed_dependence_reporting_does_not_change_raw_search():
    from prescribed_sip.methods.exchange_search import run_exchange
    from prescribed_sip.core.objective import PairObjective
    from prescribed_sip.methods.notebook_directed import directed_repetition
    demo = execute_cells({'TARGET_EXCHANGES': 15, 'FIP_TYPE': 'unequal'}, through='target')
    initial = demo['extract_support'](demo['p0'])
    raw = run_exchange(initial, PairObjective(12, demo['Pi_star']), demo['pi'],
                       seed=981, evaluations=300, seconds=100.,
                       method='GFS-Directed', batch=50, epsilon='optimal')
    reported = directed_repetition(1, 981, initial, demo['Pi_star'], demo['pi'],
                                   300, 50, 100.)[2]
    assert raw.support.mass == reported.support.mass
    assert raw.candidate_moves == reported.candidate_moves
    assert raw.accepted_moves == reported.accepted_moves
    for key in ['D2', 'target_RMSE2', 'Dinf']:
        assert raw.final[key] == reported.final[key]
    assert [c['incumbent_target_RMSE2'] for c in raw.checkpoints] == [
        c['incumbent_target_RMSE2'] for c in reported.checkpoints]


def test_distribution_reads_fragments_without_allocating_temporary_samples(monkeypatch):
    from graphical_sampling.structs import _MaxHeap, _Sample
    demo = execute_cells(through='helpers')
    heap = _MaxHeap()
    for probability, ids in [(.2, {0, 1}), (.3, {0, 1}), (.5, {2, 3})]:
        heap.push(_Sample(probability, frozenset(ids)))
    design = SimpleNamespace(_heaps=[heap], N=4, n=2)
    temporary_samples = []
    original_negation = _Sample.__neg__

    def counted_negation(sample):
        temporary_samples.append(sample.ids)
        return original_negation(sample)

    monkeypatch.setattr(_Sample, '__neg__', counted_negation)
    assert demo['distribution'](design) == {(0, 1): .5, (2, 3): .5}
    assert temporary_samples == [], 'Support extraction must avoid one new _Sample per fragment'
    np.testing.assert_array_equal(demo['compute_SIP'](design),
        np.array([[.5, .5, 0, 0], [.5, .5, 0, 0], [0, 0, .5, .5], [0, 0, .5, .5]]))


def test_directed_raw_rmse_early_stop_checks_initial_design_before_scoring():
    from prescribed_sip.core.design import Support
    from prescribed_sip.methods.notebook_directed import directed_repetition
    initial = Support(4, 2, {(0, 1): .5, (2, 3): .5})
    target = np.array([[.5,.5,0,0], [.5,.5,0,0], [0,0,.5,.5], [0,0,.5,.5]])
    result = directed_repetition(1, 123, initial, target, initial.fip(), 1000, 50, 10.)[2]
    assert result.candidate_moves == 0, 'Already recovered design must stop before any candidate scores'
    assert result.failure_reason == 'recovery_tolerance'
    assert result.final['target_RMSE2'] == 0.
    assert result.checkpoints[-1]['candidate_moves'] == 0


def test_directed_configurable_raw_rmse_threshold_uses_strict_less_than():
    from prescribed_sip.core.design import Support
    from prescribed_sip.core.objective import PairObjective
    from prescribed_sip.methods.notebook_directed import directed_repetition
    initial = Support(4, 2, {(0, 1): .5, (2, 3): .5})
    target = np.array([[.5,.5,0,0], [.5,.5,0,0], [0,0,.5,.5], [0,0,.5,.5]])
    target[0,1] = target[1,0] = .49
    target[2,3] = target[3,2] = .49
    target[1,2] = target[2,1] = .01
    target[0,3] = target[3,0] = .01
    initial_error = PairObjective(4, target).initialize(initial)['target_RMSE2']
    early = directed_repetition(1, 12, initial, target, initial.fip(), 100, 10, 10.,
                                rmse_stop=initial_error*2)[2]
    assert early.candidate_moves == 0 and early.failure_reason == 'recovery_tolerance'
    equality = directed_repetition(1, 12, initial, target, initial.fip(), 10, 10, 10.,
                                   rmse_stop=initial_error)[2]
    assert equality.candidate_moves > 0  # At equality the strict < condition is false.
    assert equality.details['raw_RMSE_stop'] == initial_error
