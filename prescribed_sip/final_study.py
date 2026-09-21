"""Final fixed-target study: reuse validated notebook algorithms, save every run."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from math import comb, fsum
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import shlex
import subprocess
import sys
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from prescribed_sip.core.design import Support
from prescribed_sip.core.dependence import DependenceError
from prescribed_sip.core.objective import PairObjective
from prescribed_sip.experiments.n12_targets import independent_sip, validate_design
from prescribed_sip.experiments.targets import seed_for
from prescribed_sip.methods.notebook_directed import directed_repetition, ReportedPairObjective
from prescribed_sip.methods.package_gfs import extract_support
from prescribed_sip.run_all import save_json, support_json, environment, execution_lock

REPO = Path(__file__).resolve().parents[1]
WORKING_NOTEBOOK = REPO / 'simulations/jupyters/GFS_prescribed_SIP_demo.ipynb'
FINAL_UNEQUAL_NOTEBOOK = Path('/home/bardia/Desktop/GFS_Prescribed_SIP_Final_Results/configs/code/working_notebook.ipynb')
NOTEBOOK = FINAL_UNEQUAL_NOTEBOOK if FINAL_UNEQUAL_NOTEBOOK.is_file() else WORKING_NOTEBOOK
METHODS = ['Initial design', 'Original GFS + restarts', 'GFS-Directed',
           'CP(2) — enumerated fit', 'DSD — projection/order fit', 'Exact LP/QP oracle']


def utc():
    return datetime.now(timezone.utc).isoformat()


def study_config(smoke=False, fip_type='unequal'):
    if fip_type not in {'equal', 'unequal'}:
        raise ValueError("fip_type must be 'equal' or 'unequal'")
    return dict(notebook=str(NOTEBOOK), master_seed=20260918, workers=6,
                fip_type=fip_type,
                scenarios=[dict(N=N,n=n,population_seed=12345+1000*k,
                                target_seed=240918+N) for k,(N,n) in enumerate([(12,4),(20,5),(30,5)])],
                target_exchanges=20 if smoke else 10000,
                directed_repetitions=2 if smoke else 30,
                directed_evaluations=100 if smoke else 100000, directed_batch=50,
                original_repetitions=1 if smoke else 5,
                original_evaluations=60 if smoke else 20000,
                dsd_trials=10 if smoke else 250,
                cp2_evaluations=80 if smoke else 50000, cp2_seconds=5 if smoke else 60,
                exact_seconds=5 if smoke else 60,
                enumeration_limit=200000, raw_rmse_stop=1e-10,
                rho_boundary_tol=1e-12, validation_tolerance=1e-10, smoke=smoke,
                original_N30_seconds_per_run=1800,
                runtime_measurement='method wall time under six-process independent-run load, not isolated hardware timing')


def cells(config):
    return {c['metadata'].get('tags',[''])[0]: ''.join(c['source'])
            for c in json.loads(Path(config['notebook']).read_text())['cells']
            if c['cell_type']=='code'}


def exec_cell(ns, source, tag):
    exec(compile(source, f'final_study_notebook:{tag}', 'exec'),ns)


def notebook_namespace(scenario, config, target=False):
    sources=cells(config); ns={'__name__':'__final_study_notebook__'}
    exec_cell(ns,sources['settings'],'settings')
    ns.update(POPULATION='synthetic',N=scenario['N'],n=scenario['n'],
              FIP_TYPE=config.get('fip_type', 'unequal'),
              SEED=scenario['population_seed'],TARGET_EXCHANGES=config['target_exchanges'],
              ALLOW_LONG_RUN=True,ENUMERATION_LIMIT=config['enumeration_limit'],
              RHO_BOUNDARY_TOL=config['rho_boundary_tol'],GFS_AUTO_RESTART=True,
              GFS_MAX_EVALUATIONS=config['original_evaluations'], GFS_RMSE_STOP=config['raw_rmse_stop'],
              DSD_MAX_EVALUATIONS=config['dsd_trials'])
    exec_cell(ns,sources['setup'],'setup')
    ns['display']=lambda *args,**kwargs:None
    ns['plt'].show=lambda:ns['plt'].close('all')
    for tag in ['population','helpers','initial']:
        exec_cell(ns,sources[tag],tag)
    if target:
        ns['SEED']=scenario['target_seed']
        for tag in ['target_order','target']:
            exec_cell(ns,sources[tag],tag)
    return ns


def name(scenario):
    return f"N{scenario['N']}_n{scenario['n']}"


def metrics(support,pi,target,initial,config):
    # Explicit sums over FINAL sample probabilities, not the optimizer cache.
    sip=independent_sip(support)
    validation=validate_design(support,pi,sip)
    validation.update(probability_sum=fsum(support.mass.values()),
                      support_bound_ok=len(support.mass)<=comb(support.N,support.n),
                      FIP_max_error=float(np.max(abs(np.diag(sip)-pi))),
                      sum_pi_error=abs(float(np.sum(pi))-support.n),
                      max_symmetry_error=float(np.max(abs(sip-sip.T))),
                      max_equal_FIP_error=(float(np.max(abs(pi-support.n/support.N)))
                                           if config.get('fip_type', 'unequal')=='equal' else None))
    if not validation['support_bound_ok'] or validation['FIP_max_error']>config['validation_tolerance']:
        raise RuntimeError(f'Independent final-design validation failed: {validation}')
    if validation['sum_pi_error']>config['validation_tolerance'] or validation['max_symmetry_error']>config['validation_tolerance']:
        raise RuntimeError(f'Independent inclusion-probability validation failed: {validation}')
    if validation['max_equal_FIP_error'] is not None and validation['max_equal_FIP_error']>config['validation_tolerance']:
        raise RuntimeError(f'Equal-FIP validation failed: {validation}')
    pair=np.triu_indices(support.N,1); residual=(sip-target)[pair]
    dependence=DependenceError(pi,config['rho_boundary_tol'])
    rho=dependence.from_matrices(sip,target)
    initial_rho=dependence.from_matrices(initial,target)
    values={'RMSE':float(np.sqrt(np.mean(residual**2))), 'RMSE_rho':rho,
            'Recovery_rho':dependence.recovery(rho,initial_rho),
            'Max SIP error':float(np.max(abs(residual))),
            'FIP max error':validation['FIP_max_error'],
            'Support size':len(support.mass),'Entropy':support.entropy(),
            **dependence.counts()}
    return values,validation,sip


def prepare_scenario(root,scenario,config):
    root=Path(root);folder=root/'targets'/name(scenario);folder.mkdir(parents=True,exist_ok=True)
    metadata=folder/'metadata.json'
    if metadata.exists():
        result=json.loads(metadata.read_text())
        if result['target_exchanges_requested']!=config['target_exchanges']:
            raise RuntimeError('Frozen target settings differ; refusing to overwrite.')
        if result.get('FIP_type', 'unequal')!=config.get('fip_type', 'unequal'):
            raise RuntimeError('Frozen target FIP type differs; refusing to overwrite.')
        for filename,digest in result['SHA256'].items():
            if hashlib.sha256((folder/filename).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Frozen target checksum mismatch')
        return result
    N,n=scenario['N'],scenario['n']
    assert N*(N-1)//2=={12:66,20:190,30:435}[N]
    folder.parent.mkdir(parents=True,exist_ok=True)
    with (folder/'target_execution.log').open('w') as log, redirect_stdout(log):
        ns=notebook_namespace(scenario,config,target=True)
    assert ns['number_of_pairs']==N*(N-1)//2
    assert len(ns['pi'])==N and abs(ns['pi'].sum()-n)<1e-10
    p0=extract_support(ns['p0']); target_design=extract_support(ns['p_star'])
    values,checks,target=metrics(target_design,ns['pi'],ns['Pi_star'],ns['Pi0'],config)
    np.testing.assert_allclose(target,ns['Pi_star'],atol=1e-12,rtol=0)
    initial_values,initial_checks,initial=metrics(p0,ns['pi'],ns['Pi_star'],ns['Pi0'],config)
    for filename,array in [('pi.npy',ns['pi']),('x.npy',ns['x']),('coords.npy',ns['coords']),
                           ('permutation.npy',ns['target_permutation']),('Pi_star.npy',ns['Pi_star']),
                           ('initial_Pi.npy',initial)]:
        np.save(folder/filename,array)
    save_json(folder/'initial_design.json',support_json(p0))
    # Only target audit/postprocessing reads this; reconstruction worker never loads it.
    save_json(folder/'target_design_AUDIT_ONLY.json',support_json(target_design))
    ns['target_history'].to_csv(folder/'target_history.csv',index=False)
    result={**scenario,'scenario':name(scenario),'FIP_type':config.get('fip_type', 'unequal'),
            'sip_pairs':N*(N-1)//2,'sample_count':comb(N,n),
            'sum_pi':float(ns['pi'].sum()),'min_pi':float(ns['pi'].min()),'max_pi':float(ns['pi'].max()),
            'accepted_exchanges':ns['accepted_exchanges'],'attempted_exchanges':ns['attempted_exchanges'],
            'target_exchanges_requested':config['target_exchanges'],
            'permutation_seed':scenario['target_seed'],'exchange_walk_seed':scenario['target_seed']+1,
            'TARGET_ALPHA':ns['TARGET_ALPHA'],'TARGET_M':ns['TARGET_M'],
            'target_procedure':'standard GFS on permuted original IDs, random package Design.exchange, exactly requested accepted non-null moves, endpoint restored to original ID order',
            'reconstruction_start':'standard original-order fixed-size package GFS design',
            'initial_to_target_RMSE':initial_values['RMSE'],
            'initial_to_target_RMSE_rho':initial_values['RMSE_rho'],
            'initial_max_SIP_discrepancy':initial_values['Max SIP error'],
            'target_runtime_seconds':ns['target_runtime'],'target_validation':checks,
            'initial_validation':initial_checks,'created_UTC':utc()}
    result['SHA256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.npy')}
    save_json(metadata,result)
    print(f"Frozen {name(scenario)}: sum(pi)={result['sum_pi']:.15g}; min/max={result['min_pi']:.8g}/{result['max_pi']:.8g}; pairs={result['sip_pairs']}; samples={result['sample_count']}; {result['accepted_exchanges']} accepted moves.",flush=True)
    return result


def read_support(path):
    d=json.loads(Path(path).read_text())
    return Support(d['N'],d['n'],dict(zip(map(tuple,d['samples']),d['probabilities'])))


def task_id(scenario,method,repetition):
    return f"{name(scenario)}_{METHODS.index(method)}_r{repetition:02d}"


def all_tasks(config):
    for scenario in config['scenarios']:
        for method in METHODS:
            repetitions=(config['directed_repetitions'] if method=='GFS-Directed' else
                         config['original_repetitions'] if method=='Original GFS + restarts' else 1)
            for repetition in range(1,repetitions+1):
                yield dict(scenario=scenario,method=method,repetition=repetition,
                           seed=seed_for(config['master_seed'],'Final',name(scenario),method,repetition))


def load_worker_problem(root,scenario,config):
    folder=Path(root)/'targets'/name(scenario)
    pi=np.load(folder/'pi.npy');target=np.load(folder/'Pi_star.npy');initial=np.load(folder/'initial_Pi.npy')
    initial_support=read_support(folder/'initial_design.json')
    assert initial_support.N==scenario['N'] and initial_support.n==scenario['n']
    return pi,target,initial,initial_support


def reconstruction_task(root,task,config):
    scenario,method,repetition,seed=task['scenario'],task['method'],task['repetition'],task['seed']
    identifier=task_id(scenario,method,repetition);root=Path(root)
    result_path=root/'raw_runs'/identifier/'result.json'
    if result_path.exists():return json.loads(result_path.read_text())
    folder=result_path.parent;folder.mkdir(parents=True,exist_ok=True)
    N,n=scenario['N'],scenario['n'];pi,target,initial,p0=load_worker_problem(root,scenario,config)
    support=None;trace=pd.DataFrame();status='completed';reason='';details={};runtime=0.;evaluations=0
    started=perf_counter()
    try:
        with (folder/'execution.log').open('w') as log,redirect_stdout(log):
            if method=='Initial design':support=p0
            elif method=='GFS-Directed':
                outcome=directed_repetition(repetition,seed,p0,target,pi,config['directed_evaluations'],
                                           config['directed_batch'],float('inf'),
                                           config['rho_boundary_tol'],config['raw_rmse_stop'],stop_inclusive=True)[2]
                support=outcome.support;runtime=outcome.runtime_seconds;evaluations=outcome.candidate_moves
                status=outcome.status;reason=outcome.failure_reason;details=outcome.details
                trace=pd.DataFrame(outcome.checkpoints)
                for output,input_column in [('evaluations','candidate_moves'),('runtime','runtime_seconds'),
                    ('RMSE','incumbent_target_RMSE2'),('RMSE_rho','incumbent_RMSE_rho'),
                    ('Recovery_rho','incumbent_Recovery_rho'),('Max SIP error','incumbent_Dinf'),
                    ('Support size','incumbent_support_size'),('Entropy','incumbent_entropy')]:
                    trace[output]=trace[input_column]
            elif method=='Original GFS + restarts':
                ns=notebook_namespace(scenario,config,target=False)
                assert 'p_star' not in ns
                np.testing.assert_array_equal(ns['pi'],pi);np.testing.assert_array_equal(ns['Pi0'],initial)
                ns['Pi_star']=target
                src=cells(config)
                exec_cell(ns,src['target_plots'].split('pair_difference =')[0],'metric_helpers')
                # Explicitly requested final-study <= rule; same optimizer/restart policy.
                adapted=src['search_helpers'].replace('best_loss < number_of_pairs * GFS_RMSE_STOP ** 2',
                                                      'best_loss <= number_of_pairs * GFS_RMSE_STOP ** 2')
                adapted=adapted.replace('D2 < number_of_pairs * GFS_RMSE_STOP ** 2',
                                        'D2 <= number_of_pairs * GFS_RMSE_STOP ** 2')
                exec_cell(ns,adapted,'search_helpers_inclusive_stopping')
                ns['SEED']=seed-2
                result=ns['run_gfs'](config['original_evaluations'],verbose=False,
                    max_seconds=config['original_N30_seconds_per_run'] if N==30 else None)
                support=extract_support(result['design']);runtime=result['runtime'];evaluations=result['evaluations']
                status=result['status'];reason=result['status'];trace=result['trace']
                details={'restarts':result['restarts'],'search_parameters':ns['search_parameters'],
                         'GFS_RESTART_EXCHANGES':ns['GFS_RESTART_EXCHANGES'],
                         'GFS_STAGNATION_EVALUATIONS':ns['GFS_STAGNATION_EVALUATIONS'],
                         'same_package_optimizer':True,'stop_rule':'raw SIP RMSE <= 1e-10'}
                result['restart_history'].to_csv(folder/'restarts.csv',index=False)
            elif method=='CP(2) — enumerated fit':
                from prescribed_sip.methods.cp2 import fit_cp2
                if comb(N,n)>config['enumeration_limit']:
                    raise MemoryError('enumeration threshold exceeded')
                outcome=fit_cp2(N,n,pi,ReportedPairObjective(p0,target,pi,config['rho_boundary_tol']),config['cp2_evaluations'],
                                config['cp2_seconds'],initial=p0)
                support=outcome.support;runtime=outcome.runtime_seconds;evaluations=outcome.candidate_moves
                status=outcome.status;reason=outcome.failure_reason;details=outcome.details
                trace=pd.DataFrame(outcome.checkpoints)
            elif method=='DSD — projection/order fit':
                ns=notebook_namespace(scenario,config,target=False);ns['Pi_star']=target
                src=cells(config)
                exec_cell(ns,src['target_plots'].split('pair_difference =')[0],'metric_helpers')
                # DSD's notebook summary uses the raw recovery helper defined here;
                # this cell defines functions but does not run package optimization.
                exec_cell(ns,src['search_helpers'],'search_helpers_for_reporting')
                exec_cell(ns,src['cp2'].split('method_results.append(summary_row')[0],'summary_helpers')
                ns['SEED']=seed-3;ns['can_enumerate']=True
                exec_cell(ns,src['dsd'],'dsd')
                support=ns['dsd_support'];runtime=ns['dsd_row']['Runtime'];evaluations=config['dsd_trials']
                details={'projection_trials':config['dsd_trials'],'analytic_to_enumerated_discrepancy':ns['discrepancies']}
            elif method=='Exact LP/QP oracle':
                from prescribed_sip.methods.exact import exact_benchmark
                result=exact_benchmark(N,n,pi,PairObjective(N,target),seconds=config['exact_seconds'])
                support=result['support'];runtime=result['runtime_seconds'];status=result['status'];reason=result['LP_message']
                details={k:v for k,v in result.items() if k!='support'}
    except (MemoryError,RuntimeError,ValueError) as error:
        status='not_run' if isinstance(error,MemoryError) else 'failed'
        reason=f'{type(error).__name__}: {error}'
        if method=='CP(2) — enumerated fit' and N==30:
            reason='CP(2) enumerated fit not run at N=30 due to enumeration/resource limit; '+reason
        support=None
    row={'Scenario':name(scenario),'N':N,'n':n,'SIP pairs':N*(N-1)//2,'Possible samples':comb(N,n),
         'Method':method,'Repetition':repetition,'Seed':seed,'Status':status,'Stop reason':reason,
         'Runtime':runtime,'Evaluations':evaluations,'Worker PID':os.getpid(),
         'Validation passed':False,'Completed UTC':utc(),
         'Target SHA256':hashlib.sha256(target.tobytes()).hexdigest(),
         'Task wall seconds':perf_counter()-started}
    if support is not None:
        values,validation,sip=metrics(support,pi,target,initial,config)
        row.update(values);row['Validation passed']=True
        save_json(folder/'design.json',support_json(support));np.save(folder/'Pi_hat.npy',sip)
        save_json(folder/'validation.json',validation)
        if method=='GFS-Directed':
            if row['RMSE']<=config['raw_rmse_stop']:
                row['Stop reason']='raw SIP RMSE <= 1e-10 (recovery_tolerance)'
            elif evaluations==config['directed_evaluations'] and not row['Stop reason']:
                row['Stop reason']='scored candidate evaluation budget reached'
            assert evaluations<=config['directed_evaluations']
        if method=='Original GFS + restarts':assert evaluations<=config['original_evaluations']
        if trace.empty:
            trace=pd.DataFrame([{'evaluations':evaluations,'runtime':runtime,**values}])
        elif 'candidate_moves' in trace:
            for output,input_column in [('evaluations','candidate_moves'),('runtime','runtime_seconds'),
                ('RMSE','incumbent_target_RMSE2'),('RMSE_rho','incumbent_RMSE_rho'),
                ('Recovery_rho','incumbent_Recovery_rho'),('Max SIP error','incumbent_Dinf'),
                ('Support size','incumbent_support_size'),('Entropy','incumbent_entropy')]:
                trace[output]=trace[input_column]
        trace.to_csv(folder/'convergence.csv',index=False)
    else:
        row.update({key:None for key in ['RMSE','RMSE_rho','Recovery_rho','Max SIP error',
                                        'FIP max error','Support size','Entropy']})
        row.update(DependenceError(pi,config['rho_boundary_tol']).counts())
    row['Peak process RSS MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    save_json(folder/'details.json',details)
    # Atomically write completion only after probabilities, SIP, trace and validation.
    save_json(result_path,row)
    return row


def save_progress(root,phase,completed,total,**more):
    save_json(Path(root)/'progress.json',{'phase':phase,'completed':completed,'total':total,
                                        'updated_UTC':utc(),'scheduler_PID':os.getpid(),**more})


def save_master(root, rows):
    path=Path(root)/'raw_runs/all_runs.csv'
    temporary=path.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(temporary,index=False,float_format='%.17g')
    temporary.replace(path)


def execute(root,config):
    root=Path(root)
    for sub in ['configs','targets','raw_runs','tables','plots','validation']:(root/sub).mkdir(parents=True,exist_ok=True)
    config_path=root/'configs/study.json'
    if config_path.exists():
        old=json.loads(config_path.read_text())
        # A resume uses exactly the previously frozen source/config.
        if old['smoke']!=config['smoke']:raise RuntimeError('Existing output uses a different study budget')
        if old.get('fip_type', 'unequal')!=config.get('fip_type', 'unequal'):
            raise RuntimeError('Existing output uses a different FIP type')
        config=old
    else:
        snapshot=root/'configs/code';snapshot.mkdir(exist_ok=True)
        copied=snapshot/'working_notebook.ipynb';shutil.copy2(config['notebook'],copied)
        config={**config,'notebook':str(copied)}
        save_json(config_path,config)
        sources=list((REPO/'graphical_sampling').rglob('*.py'))
        for relative in ['prescribed_sip','prescribed_sip/core','prescribed_sip/methods','prescribed_sip/experiments']:
            sources+=list((REPO/relative).glob('*.py'))
        for source in sources:
            dest=snapshot/source.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        save_json(root/'configs/code_SHA256.json',
            {str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in snapshot.rglob('*') if p.is_file()})
        save_json(root/'configs/environment.json',environment())
        shutil.copy2(REPO/'prescribed_sip/plans/final_study_request.txt',root/'configs/user_request.txt')
        # Frozen imports ensure a later resume cannot silently change algorithms.
        command=f"PYTHONPATH={shlex.quote(str(snapshot))} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        command+=f"MPLCONFIGDIR=/tmp/gfs-final-mpl {shlex.quote(sys.executable)} -u -m prescribed_sip.final_study --output {shlex.quote(str(root))}"
        command+=f" --fip-type {shlex.quote(config.get('fip_type', 'unequal'))}"
        if config['smoke']:command+=' --smoke'
        resume=root/'resume.sh';resume.write_text('#!/bin/sh\nset -eu\nexec env '+command+'\n');resume.chmod(0o755)
        (root/'README.txt').write_text('FINAL STUDY IN PROGRESS. See progress.json and execution.log.\n'
            'Computations are local and require no chat/API tokens. Completed runs and frozen targets are saved atomically.\n'
            'If interrupted, run ./resume.sh from this folder. It uses frozen source, targets, budgets and seeds, skips completed repetitions and reruns only unfinished repetitions.\n'
            'Do not launch a duplicate scheduler: the execution lock prevents concurrent writers.\n'
            'The final audited tables, figures and README will be generated automatically after the runs finish.\n')
    snapshot=root/'configs/code'
    for filename,digest in json.loads((root/'configs/code_SHA256.json').read_text()).items():
        if hashlib.sha256((snapshot/filename).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen code changed; refusing a scientifically inconsistent resume')
    for scenario in config['scenarios']:
        save_progress(root,'target '+name(scenario),0,len(config['scenarios']))
        prepare_scenario(root,scenario,config)
    tasks=list(all_tasks(config));rows=[];pending=[]
    for task in tasks:
        path=root/'raw_runs'/task_id(task['scenario'],task['method'],task['repetition'])/'result.json'
        if path.exists():rows.append(json.loads(path.read_text()))
        else:pending.append(task)
    save_json(root/'configs/reconstruction_seeds.json',tasks)
    print(f"Starting {len(pending)} independent runs; {config['workers']} processes; one frozen target per scenario.",flush=True)
    save_progress(root,'reconstruction',len(rows),len(tasks))
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=mp.get_context('spawn')) as pool:
        # Scenario order is deterministic; completed repetitions are never resubmitted.
        futures={pool.submit(reconstruction_task,str(root),task,config):task for task in pending}
        for future in as_completed(futures):
            task=futures[future]
            row=future.result();rows.append(row)
            save_master(root,rows)
            save_progress(root,'reconstruction',len(rows),len(tasks),last_run=row['Scenario']+' '+row['Method'],
                          last_repetition=row['Repetition'])
            print(f"{len(rows)}/{len(tasks)} saved: {row['Scenario']} {row['Method']} r{row['Repetition']}; "
                  f"RMSE={row['RMSE']}; status={row['Status']}",flush=True)
    from prescribed_sip.final_artifacts import generate_artifacts
    save_progress(root,'reporting and independent audit',len(rows),len(tasks))
    generate_artifacts(root,config)
    save_progress(root,'complete',len(rows),len(tasks))
    print('FINAL STUDY COMPLETE:',root,flush=True)


def parse_args(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--fip-type',choices=['equal','unequal'],default='unequal')
    return parser.parse_args(argv)


def main():
    args=parse_args();os.chdir(REPO)
    with execution_lock(args.output):
        execute(args.output,study_config(args.smoke,fip_type=args.fip_type))


if __name__=='__main__':main()
