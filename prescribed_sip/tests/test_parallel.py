"""Regression tests for scheduling, immutable resume and execution provenance."""
from pathlib import Path
import hashlib
import importlib.util
import importlib.machinery
import json
import os
import pytest
import pandas as pd
from prescribed_sip.run_all import execute


def configuration(root,regimes=('evaluations',),replications=3):
    return {'master_seed':902,'output':str(root),
            'methods':['Package-GFS','GFS-Random','GFS-Greedy','GFS-Directed'],
            'budget':{'evaluations':40,'seconds':20,'batch':10},
            'scenarios':[{'experiment':'small','N':6,'n':2,'target_type':'srs','geometry':'1d',
                          'replications':replications,'regimes':list(regimes)}]}


def saved_rows(root):
    return {p.name:json.loads(p.read_text())['row'] for p in (root/'results/runs').glob('*.json')}


def test_parallel_matches_frozen_serial_runner_and_uses_child_processes(tmp_path):
    # Catches changed seeds, missing jobs, shared RNG state or replaced methods.
    old=Path('prescribed_sip/reports/source_snapshots/pre_parallel/prescribed_sip/run_all.py.txt')
    loader=importlib.machinery.SourceFileLoader('legacy_serial_runner',str(old))
    spec=importlib.util.spec_from_loader(loader.name,loader);legacy=importlib.util.module_from_spec(spec);loader.exec_module(legacy)
    serial=tmp_path/'serial';parallel=tmp_path/'parallel'
    legacy.execute(configuration(serial),artifacts=False)
    execute(configuration(parallel),artifacts=False,workers=2)
    a,b=saved_rows(serial),saved_rows(parallel)
    assert len(a)==15 and a.keys()==b.keys()
    for name in a:
        for key in ['seed','target_seed','initial_seed','candidate_moves','accepted_moves','final_support_size','status']:
            assert a[name][key]==b[name][key],(name,key)
        for key in ['final_D2','final_target_RMSE2','max_FIP_error','max_probability_sum_error']:
            assert b[name][key]==pytest.approx(a[name][key],abs=1e-13)
        assert b[name]['execution_workers']==2
        assert b[name]['worker_process_id']!=os.getpid()
    assert json.loads((parallel/'results/progress.json').read_text())['status']=='completed'


def test_timing_runs_stay_in_parent_and_existing_json_is_immutable(tmp_path):
    # Catches timed tasks entering the worker pool or resume rewriting saved rows.
    config=configuration(tmp_path,regimes=('evaluations','time'),replications=2)
    config['budget']['seconds']=.1
    execute(config,artifacts=False,workers=2)
    rows=saved_rows(tmp_path)
    assert len(rows)==18
    for row in rows.values():
        if row['regime']=='time':
            assert row['execution_workers']==1
            assert row['worker_process_id']==os.getpid()
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (tmp_path/'results/runs').glob('*.json')}
    execute(config,artifacts=False,workers=2)
    after={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (tmp_path/'results/runs').glob('*.json')}
    assert before==after


def legacy_manifest(root,snapshot):
    config=configuration(root,replications=1)
    execute(config,artifacts=False)
    path=root/'results/manifest.json';m=json.loads(path.read_text())
    old=b'Previous execution runner, preserved for migration audit.\n'
    m['environment']['simulation_source_SHA256']['prescribed_sip/run_all.py']=hashlib.sha256(old).hexdigest()
    target=snapshot/'prescribed_sip/run_all.py.txt';target.parent.mkdir(parents=True);target.write_bytes(old)
    path.write_text(json.dumps(m))
    return config


def test_execution_migration_requires_verified_snapshot_and_explicit_flag(tmp_path):
    # Catches silent mixing of different executables or overwriting legacy evidence.
    root=tmp_path/'output';snapshot=tmp_path/'source'
    config=legacy_manifest(root,snapshot)
    before={p.name:p.read_bytes() for p in (root/'results/runs').glob('*.json')}
    with pytest.raises(ValueError,match='source changed'):
        execute(config,artifacts=False,workers=2)
    execute(config,artifacts=False,workers=2,allow_execution_change=True,migration_snapshot=snapshot)
    m=json.loads((root/'results/manifest.json').read_text())
    assert m['environment']['simulation_source_SHA256']['prescribed_sip/run_all.py']==hashlib.sha256((snapshot/'prescribed_sip/run_all.py.txt').read_bytes()).hexdigest()
    assert len(m['execution_history'])==1
    assert before=={p.name:p.read_bytes() for p in (root/'results/runs').glob('*.json')}


def test_migration_refuses_numerical_source_change(tmp_path):
    # Catches allowing a changed optimizer/target formula under a scheduling flag.
    root=tmp_path/'output';snapshot=tmp_path/'source';config=legacy_manifest(root,snapshot)
    p=root/'results/manifest.json';m=json.loads(p.read_text())
    m['environment']['simulation_source_SHA256']['prescribed_sip/core/objective.py']='0'*64
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='scientific'):
        execute(config,artifacts=False,workers=2,allow_execution_change=True,migration_snapshot=snapshot)


def test_migration_refuses_unverified_old_execution_source(tmp_path):
    root=tmp_path/'output';snapshot=tmp_path/'source';config=legacy_manifest(root,snapshot)
    (snapshot/'prescribed_sip/run_all.py.txt').write_text('wrong backup')
    with pytest.raises(ValueError,match='snapshot'):
        execute(config,artifacts=False,workers=2,allow_execution_change=True,migration_snapshot=snapshot)


def test_legacy_concurrency_is_defaulted_without_rewriting_json(tmp_path):
    from prescribed_sip.run_all import collect
    folder=tmp_path/'results/runs';folder.mkdir(parents=True)
    p=folder/'old.json';p.write_text(json.dumps({'row':{'seed':1},'checkpoints':[]}));before=p.read_bytes()
    collect(tmp_path)
    assert pd.read_csv(tmp_path/'results/runs.csv').execution_workers.tolist()==[1]
    assert p.read_bytes()==before


def test_summaries_do_not_pool_different_execution_worker_limits():
    # Catches mixing timings or safety-capped accuracy across serial/parallel loads.
    from prescribed_sip.run_all import base_row
    from prescribed_sip.experiments.artifacts import summarize,paired
    scenario={'experiment':'small','N':6,'n':2,'target_type':'srs'}
    first=base_row(scenario,'s',0,'Package-GFS','default',1,{'evaluations':40,'seconds':1})
    first.update({'status':'completed','feasible':True,'final_RMSE2':.1,'final_target_RMSE2':.1,'execution_workers':1})
    second={**first,'final_RMSE2':.2,'final_target_RMSE2':.2,'execution_workers':4}
    competitor={**first,'method':'GFS-Greedy','final_RMSE2':.05,'final_target_RMSE2':.05,'execution_workers':4}
    raw=pd.DataFrame([first,second,competitor])
    summary=summarize(raw,1)
    assert len(summary)==3
    comparison=paired(raw,1)
    assert len(comparison)==1
    assert comparison.iloc[0].complete_pairs==1
    assert comparison.iloc[0].mean_competitor_minus_primary==pytest.approx(-.15)


def test_exact_only_legacy_partial_case_is_completed_serially(tmp_path):
    # Catches a legacy benchmark being overlooked before the first method saved.
    config=configuration(tmp_path,replications=1)
    execute(config,artifacts=False)
    folder=tmp_path/'results/runs';exact=next(folder.glob('*_exact.json'));before=exact.read_bytes()
    for path in folder.glob('*.json'):
        if path!=exact:path.unlink()
    execute(config,artifacts=False,workers=2)
    rows=saved_rows(tmp_path)
    assert len(rows)==5 and exact.read_bytes()==before
    assert all(row['execution_workers']==1 for row in rows.values())
    assert all(row['worker_process_id']==os.getpid() for row in rows.values())


def test_duplicate_study_writer_is_rejected_before_creating_manifest(tmp_path):
    # Catches two schedulers writing the same study output concurrently.
    import prescribed_sip.run_all as runner
    lock=getattr(runner,'execution_lock',None)
    assert lock is not None,'runner must own an exclusive execution lock'
    config=configuration(tmp_path,replications=1)
    with lock(tmp_path):
        with pytest.raises(RuntimeError,match='already running'):
            execute(config,artifacts=False,workers=2)
    assert not (tmp_path/'results/manifest.json').exists()


def test_pending_legacy_timing_is_deferred_until_after_evaluation_pool(tmp_path,capsys):
    # Catches serial timing backlogs delaying the parallel evaluation phase.
    config=configuration(tmp_path,regimes=('evaluations','time'),replications=2)
    config['budget']['seconds']=.1
    execute(config,artifacts=False)
    folder=tmp_path/'results/runs'
    for path in folder.glob('*.json'):
        row=json.loads(path.read_text())['row']
        if row['replication']==1 or row['regime']=='time':path.unlink()
    capsys.readouterr()
    execute(config,artifacts=False,workers=2)
    out=capsys.readouterr().out
    assert out.index('Evaluation phase:')<out.index('[time; workers=1]')
    rows=saved_rows(tmp_path)
    assert len(rows)==18
    assert all(row['execution_workers']==1 for row in rows.values() if row['regime']=='time')
