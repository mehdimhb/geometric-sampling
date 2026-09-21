"""Final driver must reuse the working notebook and freeze original-ID targets."""
from pathlib import Path
import numpy as np


def test_equal_final_config_changes_only_fip_and_builds_constant_vectors():
    import hashlib
    from prescribed_sip.final_study import notebook_namespace, study_config

    unequal = study_config(smoke=True)
    equal = study_config(smoke=True, fip_type='equal')
    assert unequal['fip_type'] == 'unequal'
    assert equal['fip_type'] == 'equal'
    assert {key: value for key, value in unequal.items() if key != 'fip_type'} == {
        key: value for key, value in equal.items() if key != 'fip_type'
    }
    frozen_notebook = Path(
        '/home/bardia/Desktop/GFS_Prescribed_SIP_Final_Results/configs/code/working_notebook.ipynb'
    )
    assert hashlib.sha256(Path(equal['notebook']).read_bytes()).digest() == hashlib.sha256(
        frozen_notebook.read_bytes()
    ).digest()
    for scenario in equal['scenarios']:
        namespace = notebook_namespace(scenario, equal, target=False)
        np.testing.assert_array_equal(
            namespace['pi'],
            np.full(scenario['N'], scenario['n'] / scenario['N']),
        )


def test_final_study_cli_accepts_explicit_equal_fip(tmp_path):
    from prescribed_sip.final_study import parse_args

    args = parse_args(['--output', str(tmp_path), '--fip-type', 'equal', '--smoke'])
    assert args.output == tmp_path
    assert args.fip_type == 'equal'
    assert args.smoke is True


def test_equal_target_metadata_records_equal_fip_and_valid_target(tmp_path):
    from prescribed_sip.final_study import prepare_scenario, study_config

    config = study_config(smoke=True, fip_type='equal')
    config['target_exchanges'] = 3
    scenario = config['scenarios'][0]
    metadata = prepare_scenario(tmp_path, scenario, config)
    pi = np.load(tmp_path / 'targets/N12_n4/pi.npy')
    target = np.load(tmp_path / 'targets/N12_n4/Pi_star.npy')

    assert metadata['FIP_type'] == 'equal'
    assert metadata['target_validation']['sum_pi_error'] == 0
    assert metadata['target_validation']['max_equal_FIP_error'] == 0
    assert metadata['target_validation']['max_symmetry_error'] == 0
    np.testing.assert_array_equal(pi, np.full(12, 1 / 3))
    np.testing.assert_allclose(target, target.T, atol=0, rtol=0)
    np.testing.assert_allclose(np.diag(target), pi, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        target.sum(axis=1) - np.diag(target), 3 * pi, atol=1e-12, rtol=0
    )


def test_requested_equal_publication_table_uses_raw_recovery_and_exact_layout():
    import pandas as pd
    from prescribed_sip.final_artifacts import requested_publication_table

    full = pd.DataFrame([
        {'N': 12, 'n': 4, 'Method': 'Initial design', 'Repetitions': 1,
         'RMSE mean': 0.2, 'RMSE SD': np.nan, 'Max SIP error': 0.4,
         'Runtime mean': 0.0, 'Runtime SD': np.nan,
         'Evaluations mean': 0.0, 'Evaluations SD': np.nan,
         'Support size mean': 12.0, 'Support size SD': np.nan},
        {'N': 12, 'n': 4, 'Method': 'GFS-Directed', 'Repetitions': 2,
         'RMSE mean': 0.05, 'RMSE SD': 0.01, 'Max SIP error': 0.1,
         'Runtime mean': 2.0, 'Runtime SD': 0.5,
         'Evaluations mean': 100.0, 'Evaluations SD': 5.0,
         'Support size mean': 20.0, 'Support size SD': 2.0},
    ])

    table = requested_publication_table(full)
    assert list(table.columns) == [
        'N', 'n', 'Method', 'RMSE_pi2', 'Recovery (%)', 'Max SIP error',
        'Time (s)', 'Eval.', 'Support',
    ]
    directed = table.loc[table.Method == 'GFS-Directed'].iloc[0]
    assert directed['RMSE_pi2'] == '5.000e-02 ± 1.000e-02'
    assert directed['Recovery (%)'] == '75.00 ± 5.00'


def test_equal_target_exports_and_configuration_audit(tmp_path):
    import json
    from prescribed_sip.final_artifacts import export_equal_targets, write_comparability_audit
    from prescribed_sip.final_study import study_config

    equal = study_config(smoke=True, fip_type='equal')
    unequal = study_config(smoke=True, fip_type='unequal')
    for scenario in equal['scenarios']:
        folder = tmp_path / 'targets' / f"N{scenario['N']}_n{scenario['n']}"
        folder.mkdir(parents=True)
        pi = np.full(scenario['N'], scenario['n'] / scenario['N'])
        np.save(folder / 'pi.npy', pi)
        np.save(folder / 'Pi_star.npy', np.diag(pi))

    export_equal_targets(tmp_path, equal)
    for scenario in equal['scenarios']:
        label = f"N{scenario['N']}_n{scenario['n']}"
        np.testing.assert_array_equal(
            np.loadtxt(tmp_path / f'{label}_fip.csv', delimiter=','),
            np.full(scenario['N'], scenario['n'] / scenario['N']),
        )
        assert (tmp_path / f'{label}_target_sip.csv').exists()
    combined = np.load(tmp_path / 'final_equal_targets.npz')
    assert set(combined.files) == {
        'pi_N12', 'Pi_star_N12', 'pi_N20', 'Pi_star_N20',
        'pi_N30', 'Pi_star_N30',
    }

    audit = write_comparability_audit(tmp_path, equal, unequal)
    assert audit['parameters_identical'] is True
    assert audit['parameter_differences'] == {'fip_type': ['unequal', 'equal']}
    text = (tmp_path / 'COMPARABILITY_AUDIT.txt').read_text()
    assert 'Parameters identical to unequal-FIP experiment: YES' in text
    assert 'FIP_TYPE: unequal -> equal' in text


def test_final_study_reuses_notebook_target_and_original_start(tmp_path):
    from prescribed_sip.final_study import prepare_scenario, study_config, notebook_namespace
    config = study_config(smoke=True)
    config['target_exchanges'] = 20
    scenario = config['scenarios'][0]
    problem = prepare_scenario(tmp_path, scenario, config)
    assert problem['N'] == 12 and problem['n'] == 4
    assert problem['sip_pairs'] == 66 and problem['sample_count'] == 495
    assert problem['accepted_exchanges'] == 20
    ns = notebook_namespace(scenario, config, target=True)
    np.testing.assert_array_equal(np.load(tmp_path / 'targets/N12_n4/Pi_star.npy'), ns['Pi_star'])
    assert 'p_star' not in notebook_namespace(scenario, config, target=False)
    np.testing.assert_array_equal(ns['p0'].order.get()[:, 0].astype(int), np.arange(12))
    np.testing.assert_allclose(np.diag(ns['Pi_star']), ns['pi'], atol=1e-12)


def test_final_directed_inclusive_stop_handles_exact_zero_without_candidates():
    from prescribed_sip.core.design import Support
    from prescribed_sip.experiments.n12_targets import independent_sip
    from prescribed_sip.methods.notebook_directed import directed_repetition
    initial = Support(3, 2, {(0, 1): 1/3, (0, 2): 1/3, (1, 2): 1/3})
    outcome = directed_repetition(1, 42, initial, independent_sip(initial),
                                  np.full(3, 2/3), 100, 50, 10,
                                  rmse_stop=0, stop_inclusive=True)[2]
    assert outcome.candidate_moves == 0
    assert outcome.failure_reason == 'recovery_tolerance'


def test_final_summary_uses_sample_sd_and_never_invents_single_run_sd():
    import pandas as pd
    from prescribed_sip.final_artifacts import summarize
    rows = pd.DataFrame([
        {'N':12, 'n':4, 'Method':'GFS-Directed', 'Validation passed':True,
         'RMSE':value, 'RMSE_rho':value*2, 'Recovery_rho':100-value*10,
         'Max SIP error':value*3, 'FIP max error':1e-15,
         'Runtime':value, 'Evaluations':100, 'Support size':10,
         'Entropy':value, 'Rho pairs included':66, 'Rho pairs excluded':0}
        for value in [1.,3.]])
    table = summarize(rows)
    row = table.loc[table['Method']=='GFS-Directed'].iloc[0]
    assert row['Repetitions']==2
    assert row['RMSE mean']==2 and row['RMSE median']==2
    np.testing.assert_allclose(row['RMSE SD'], np.sqrt(2))
    single = summarize(rows.iloc[:1])
    assert pd.isna(single.loc[single['Method']=='GFS-Directed', 'RMSE SD'].iloc[0])


def test_final_small_execution_exports_and_audits_real_designs(tmp_path):
    import json
    import pandas as pd
    from prescribed_sip.final_study import execute, study_config
    config = study_config(smoke=True)
    config['scenarios'] = config['scenarios'][:1]
    config['workers'] = 2
    execute(tmp_path, config)
    assert json.loads((tmp_path/'progress.json').read_text())['phase']=='complete'
    runs = pd.read_csv(tmp_path/'raw_runs/all_runs.csv')
    assert len(runs)==7 and runs['Validation passed'].all()
    assert runs.loc[runs.Method=='GFS-Directed', 'Seed'].nunique()==2
    assert runs['SIP pairs'].eq(66).all()
    assert (tmp_path/'tables/Table_1_N12_n4.tex').exists()
    assert (tmp_path/'tables/Table_All_Scenarios.csv').exists()
    assert len(list((tmp_path/'plots/N12_n4').glob('*.pdf')))>=7
    assert len(list((tmp_path/'plots/N12_n4').glob('*.png')))>=7
    audit = pd.read_csv(tmp_path/'validation/validation.csv')
    assert audit['Passed'].all()
    # Exercise the user-facing resume command, including frozen module imports.
    import subprocess
    resumed = subprocess.run(['sh', str(tmp_path/'resume.sh')],
                             capture_output=True, text=True, timeout=60)
    assert resumed.returncode==0, resumed.stderr
    assert 'Starting 0 independent runs' in resumed.stdout
    pd.testing.assert_frame_equal(runs,pd.read_csv(tmp_path/'raw_runs/all_runs.csv'))


def test_equal_small_execution_writes_requested_outputs(tmp_path):
    import pandas as pd
    from prescribed_sip.final_study import execute, study_config

    config = study_config(smoke=True, fip_type='equal')
    config['scenarios'] = config['scenarios'][:1]
    config['workers'] = 2
    execute(tmp_path, config)

    requested = pd.read_csv(tmp_path / 'tables/Equal_FIP_Results.csv')
    assert list(requested.columns) == [
        'N', 'n', 'Method', 'RMSE_pi2', 'Recovery (%)', 'Max SIP error',
        'Time (s)', 'Eval.', 'Support',
    ]
    assert (tmp_path / 'tables/Equal_FIP_Results.tex').exists()
    assert (tmp_path / 'tables/Equal_FIP_Results.md').exists()
    assert (tmp_path / 'N12_n4_fip.csv').exists()
    assert (tmp_path / 'N12_n4_target_sip.csv').exists()
    assert set(np.load(tmp_path / 'final_equal_targets.npz').files) == {
        'pi_N12', 'Pi_star_N12',
    }
    assert 'Scenarios: equal pi_i = n/N.' in (tmp_path / 'README.txt').read_text()
    assert '--fip-type equal' in (tmp_path / 'resume.sh').read_text()
