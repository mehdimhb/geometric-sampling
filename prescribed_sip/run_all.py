"""Serial, resumable staged runner. Run from the repository root."""
import argparse
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import subprocess
import sys
import hashlib
import importlib.metadata
import resource
import numpy as np
import pandas as pd
import yaml
from prescribed_sip.core.objective import PairObjective
from prescribed_sip.core.design import Support,InvariantError
from prescribed_sip.experiments.targets import make_target,seed_for
from prescribed_sip.methods.package_gfs import seeded_package,initial_design,extract_support,run_package
from prescribed_sip.methods.exchange_search import run_exchange
from prescribed_sip.methods.exact import exact_benchmark
from prescribed_sip.methods.cp2 import fit_cp2
from prescribed_sip.methods.benchmarks import structured_outcome


PRIMARY='Package-GFS'


def json_default(obj):
    if isinstance(obj,np.generic): return obj.item()
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,Path): return str(obj)
    raise TypeError(type(obj).__name__)


def save_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,default=json_default,indent=2)+'\n'); temp.replace(path)


def support_json(s):
    return {'N':s.N,'n':s.n,'samples':[list(v) for v in s.mass],
            'probabilities':list(s.mass.values())}


def environment():
    versions={}
    for name in ['numpy','scipy','matplotlib','pandas','pytest','osqp','PyYAML','graphical-sampling','package-sampling']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='unavailable'
    original_files=['graphical_sampling/design.py','graphical_sampling/order.py','graphical_sampling/structs.py',
                    'graphical_sampling/criteria/__init__.py','graphical_sampling/criteria/criteria.py',
                    'graphical_sampling/search/gbfs.py']
    return {'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
            'CPU_count':os.cpu_count(),'dependencies':versions,
            'git_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
            'git_status':subprocess.run(['git','status','--short'],capture_output=True,text=True).stdout,
            'original_source_SHA256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in original_files},
            'simulation_source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('prescribed_sip').rglob('*.py')},
            'thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}}


def expand_scenarios(config):
    if 'scenarios' in config:
        return config['scenarios']
    result=[]
    for experiment,stage in config['stages'].items():
        for N,n in stage['sizes']:
            for kind in stage['targets']:
                for geometry in stage.get('geometries',['planar','1d']):
                    for density in stage.get('densities',[1.]):
                        structures=stage.get('pair_structures',['random']) if density<1 else ['random']
                        for structure in structures:
                            result.append({'experiment':experiment,'N':N,'n':n,'target_type':kind,'geometry':geometry,
                                           'density':density,'pair_structure':structure,'replications':stage['replications'],
                                           'budget':stage.get('budget',config['budget']),
                                           'regimes':stage.get('regimes',['evaluations'])})
    return result


def base_row(scenario,scenario_id,replication,method,variant,seed,budget):
    N=scenario['N']; n=scenario['n']
    return {'experiment':scenario['experiment'],'scenario_id':scenario_id,'target_type':scenario['target_type'],
            'N':N,'n':n,'sampling_fraction':n/N,'FIP_type':'unknown','geometry':scenario.get('geometry','planar'),
            'target_density':scenario.get('density',1.),'pair_structure':scenario.get('pair_structure','random'),
            'method':method,'method_variant':variant,'replication':replication,'seed':seed,
            'initial_D2':np.nan,'final_D2':np.nan,'initial_RMSE2':np.nan,'final_RMSE2':np.nan,
            'initial_Dinf':np.nan,'final_Dinf':np.nan,'initial_target_RMSE2':np.nan,'final_target_RMSE2':np.nan,
            'runtime_seconds':np.nan,'candidate_moves':0,'accepted_moves':0,'acceptance_rate':np.nan,
            'initial_support_size':np.nan,'final_support_size':np.nan,'entropy':np.nan,
            'max_FIP_error':np.nan,'max_probability_sum_error':np.nan,'min_sample_probability':np.nan,
            'max_SIP_incremental_error':np.nan,'converged':False,'exact_recovery':False,'timeout':False,
            'failure_reason':'','status':'unexecuted','QP_optimal_D2':np.nan,'GFS_optimality_gap':np.nan,
            'GFS_excess_over_QP':np.nan,'QP_certificate_gap':np.nan,'QP_lower_bound_D2':np.nan,
            'feasible':None,'relative_improvement':np.nan,'support_growth':np.nan,
            'regime':budget.get('regime','evaluations'),'evaluation_budget':budget['evaluations'],
            'time_budget_seconds':budget['seconds'],'candidate_batch':budget.get('batch',10),
            'peak_process_RSS_MB':np.nan,'memory_measurement':'process lifetime high-water RSS; not isolated method peak',
            'recovery_1e4':False,'recovery_1e6':False,'recovery_1e8':False}


def populate_outcome(row,outcome,initial,benchmark,target):
    for prefix,metrics in [('initial',outcome.initial),('final',outcome.final)]:
        for k,v in metrics.items(): row[prefix+'_'+k]=v
    row.update({'runtime_seconds':outcome.runtime_seconds,'candidate_moves':outcome.candidate_moves,
                'accepted_moves':outcome.accepted_moves,'acceptance_rate':outcome.accepted_moves/outcome.candidate_moves if outcome.candidate_moves else np.nan,
                'initial_support_size':len(initial.mass),'status':outcome.status,'failure_reason':outcome.failure_reason,
                'timeout':outcome.status=='timeout',**outcome.validation,
                'peak_process_RSS_MB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                'method_details':json.dumps(outcome.details,default=json_default,sort_keys=True)})
    if outcome.support is not None:
        row.update({'final_support_size':len(outcome.support.mass),'entropy':outcome.support.entropy(),
                    'support_growth':len(outcome.support.mass)/len(initial.mass)})
    row['feasible']=benchmark['feasible'] if benchmark else target.feasible_known
    if benchmark and benchmark['status'].startswith('certified'):
        row.update({'QP_optimal_D2':benchmark['optimal_D2'],'QP_certificate_gap':benchmark['certificate_gap'],
                    'QP_lower_bound_D2':benchmark['lower_bound_D2']})
        gap=row['final_D2']-benchmark['optimal_D2']
        row['GFS_optimality_gap']=gap; row['GFS_excess_over_QP']=gap
    loss0=row['initial_D2']; row['relative_improvement']=1-row['final_D2']/loss0 if loss0>0 else np.nan
    row['exact_recovery']=bool(row['feasible'] is True and row['final_target_RMSE2']<1e-8)
    row['converged']=bool(row['final_target_RMSE2']<1e-10)
    for tolerance,label in [(1e-4,'recovery_1e4'),(1e-6,'recovery_1e6'),(1e-8,'recovery_1e8')]:
        row[label]=bool(row['feasible'] is True and row['final_target_RMSE2']<tolerance)


def run_one(target,initial_package,method,settings,seed,budget):
    objective=PairObjective(target.N,target.sip); initial=extract_support(initial_package)
    options={'evaluations':budget['evaluations'],'seconds':budget['seconds'],'batch':settings.get('batch',budget.get('batch',10))}
    if method==PRIMARY:
        return run_package(initial_package,objective,target.pi,seed=seed,**options)
    if method.startswith('GFS-'):
        def initializer(s):
            with seeded_package(s):
                d=initial_design(target.pi,target.coords,np.random.default_rng().permutation(target.N))
            return extract_support(d)
        return run_exchange(initial,objective,target.pi,seed=seed,method=method,
                            epsilon=settings.get('epsilon','random' if method=='GFS-Random' else 'optimal'),
                            starts=settings.get('starts',3),initializer=initializer,**options)
    if method=='CP2-enumerated':
        if target.N>14: raise NotImplementedError('exact CP(2) partition-function implementation limited to N<=14')
        return fit_cp2(target.N,target.n,target.pi,objective,initial=initial,seed=seed,**options)
    if method in ('Sinha','DSD-structured'):
        return structured_outcome(target,objective,initial,method)
    raise NotImplementedError(f'method {method} has no reliable implemented specification')


def prepare_execution(config,allow_execution_change=False,migration_snapshot=None):
    """Validate identity; explicitly audited scheduler changes retain old evidence."""
    root=Path(config.get('output','prescribed_sip'))
    for folder in ['results/runs','results/targets','results/designs','figures','tables','reports']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest_path=root/'results/manifest.json'
    now=environment()
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['config_SHA256']!=fingerprint:
            raise ValueError('output directory belongs to another config; choose a new output')
        prior=old.get('execution_history',[])
        previous=prior[-1]['environment'] if prior else old['environment']
        source_changed=any(previous.get(k)!=now[k] for k in ['simulation_source_SHA256','original_source_SHA256'])
        if source_changed:
            if not allow_execution_change:
                raise ValueError('source changed since results were generated; use a new output directory or an audited execution migration')
            if previous['original_source_SHA256']!=now['original_source_SHA256']:
                raise ValueError('scientific original-package source changed; migration refused')
            before=previous['simulation_source_SHA256'];after=now['simulation_source_SHA256']
            changed=[p for p,h in before.items() if after.get(p)!=h]
            added=[p for p in after if p not in before]
            def execution_source(p):
                return p in {'prescribed_sip/run_all.py','prescribed_sip/parallel.py','prescribed_sip/experiments/artifacts.py'} or p.startswith('prescribed_sip/tests/')
            if any(not execution_source(p) for p in changed+added):
                raise ValueError('scientific simulation source changed; migration refused')
            snapshot=Path(migration_snapshot or 'prescribed_sip/reports/source_snapshots/pre_parallel')
            for name in changed:
                frozen=snapshot/(name+'.txt')
                if not frozen.exists() or hashlib.sha256(frozen.read_bytes()).hexdigest()!=before[name]:
                    raise ValueError(f'unverified old execution snapshot: {name}')
            old.setdefault('execution_history',[]).append({'UTC':utc_now(),'environment':now,
                'changed_execution_sources':changed,'added_execution_sources':added,
                'verified_previous_snapshot':str(snapshot),'reason':'user-approved parallel scheduling; preserved completed results'})
        old['status']='running';save_json(manifest_path,old)
    else:
        save_json(manifest_path,{'config_SHA256':fingerprint,'config':config,'environment':now,
            'primary_method':PRIMARY,'status':'running','executed_stages':[]})
    return root


def utc_now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()


def execute_replication(config,root,scenario,replication,regimes=None,execution_workers=1,progress=None):
    """Original numerical execution body; one process owns each replication."""
    root=Path(root);methods=config['methods']
    N,n=scenario['N'],scenario['n'];kind=scenario['target_type'];geom=scenario.get('geometry','planar')
    density=scenario.get('density',1.);structure=scenario.get('pair_structure','random')
    sid=f'{scenario["experiment"]}_N{N}_n{n}_{kind}_{geom}_d{density:g}_{structure}'
    target_seed=seed_for(config['master_seed'],N,n,kind,replication,'target')
    start_seed=seed_for(config['master_seed'],N,n,kind,replication,'initial')
    target_file=root/f'results/targets/{sid}_r{replication}.npz'
    target=None; package=None; initial=None; initialization_failure=''
    try:
        target=make_target(N,n,kind,target_seed,geom,density,structure)
        with seeded_package(start_seed): package=initial_design(target.pi,target.coords)
        initial=extract_support(package)
        if isinstance(target.sip,dict):
            pairs=list(target.sip); stored={'pairs':np.array(pairs),'sip_values':np.array(list(target.sip.values()))}
        else: stored={'sip':target.sip}
        if not target_file.exists():
            np.savez_compressed(target_file,pi=target.pi,coords=target.coords,**stored)
            save_json(target_file.with_suffix('.json'),target.metadata)
            save_json(root/f'results/designs/{sid}_r{replication}_initial.json',support_json(initial))
    except (ValueError,RuntimeError) as e:
        initialization_failure=f'{type(e).__name__}: {e}'
    benchmark=None
    benchmark_path=root/f'results/runs/{sid}_r{replication}_exact.json'
    if N<=14 and target is not None and not initialization_failure:
        if benchmark_path.exists():
            stored=json.loads(benchmark_path.read_text()); benchmark=stored['benchmark']
        else:
            try: benchmark=exact_benchmark(N,n,target.pi,PairObjective(N,target.sip),seconds=config.get('exact_seconds',10))
            except Exception as e:
                benchmark={'status':'solver_failure','feasible':None,'optimal_D2':np.nan,'certificate_gap':np.nan,
                           'lower_bound_D2':np.nan,'support':None,'runtime_seconds':0.,'failure_reason':str(e)}
            row=base_row(scenario,sid,replication,'Exact-LP-QP','enumerated',start_seed,scenario.get('budget',config['budget']))
            row.update({'execution_workers':execution_workers,'worker_process_id':os.getpid()})
            row.update({'target_seed':target_seed,'initial_seed':start_seed})
            row['FIP_type']=target.fip_type; row['status']=benchmark['status']; row['feasible']=benchmark['feasible']
            row['runtime_seconds']=benchmark['runtime_seconds']; row['failure_reason']=benchmark.get('failure_reason',benchmark.get('QP_status',''))
            bs=benchmark.pop('support',None)
            if bs is not None:
                obj=PairObjective(N,target.sip); initial_metrics=obj.initialize(initial); final=obj.initialize(bs)
                for pre,metrics in [('initial',initial_metrics),('final',final)]:
                    for k,v in metrics.items(): row[pre+'_'+k]=v
                row.update(bs.validate(target.pi)); row['max_SIP_incremental_error']=0.
                row['initial_support_size']=len(initial.mass); row['final_support_size']=len(bs.mass)
                row['entropy']=bs.entropy(); row['QP_optimal_D2']=benchmark['optimal_D2']
                row['QP_certificate_gap']=benchmark['certificate_gap']; row['QP_lower_bound_D2']=benchmark['lower_bound_D2']
                for tol,label in [(1e-4,'recovery_1e4'),(1e-6,'recovery_1e6'),(1e-8,'recovery_1e8')]:
                    row[label]=bool(benchmark['feasible'] is True and final['target_RMSE2']<tol)
                row['exact_recovery']=row['recovery_1e8']; row['converged']=final['target_RMSE2']<1e-10
                save_json(root/f'results/designs/{sid}_r{replication}_exact.json',support_json(bs))
            save_json(benchmark_path,{'row':row,'benchmark':benchmark,'checkpoints':[]})
    regimes=scenario.get('regimes',['evaluations']) if regimes is None else regimes
    for regime in regimes:
        default_budget=dict(scenario.get('budget',config['budget']))
        default_budget['regime']=regime
        if regime=='time': default_budget['evaluations']=config.get('time_evaluation_ceiling',10000000)
        method_entries=[]
        for item in methods:
            settings={'name':item} if isinstance(item,str) else dict(item)
            method_entries.append(settings)
        if scenario['experiment']=='ablation': method_entries=config.get('ablation_methods',method_entries)
        for settings in method_entries:
            method=settings['name']; variant=settings.get('variant','default')
            filename=f'{sid}_r{replication}_{method}_{variant}_{regime}'
            output=root/f'results/runs/{filename}.json'
            if output.exists(): continue
            if progress: progress({'scenario_id':sid,'replication':replication+1,'method':method,'regime':regime})
            method_seed=seed_for(config['master_seed'],N,n,kind,replication,geom,method,variant,regime,'method')
            row=base_row(scenario,sid,replication,method,variant,method_seed,default_budget)
            row.update({'execution_workers':execution_workers,'worker_process_id':os.getpid()})
            row.update({'target_seed':target_seed,'initial_seed':start_seed,
                        'candidate_batch':settings.get('batch',default_budget.get('batch',10))})
            if target:
                row['FIP_type']=target.fip_type
                row['feasible']=benchmark['feasible'] if benchmark else target.feasible_known
            outcome=None; started=perf_counter()
            if initialization_failure:
                row['status']='numerical_failure'; row['failure_reason']=initialization_failure
            else:
                try:
                    if method=='CP2-enumerated' and benchmark and benchmark['feasible'] is False:
                        raise NotImplementedError('exact CP(2) moment calibration target is LP-infeasible; no ad hoc D2 fit substituted')
                    outcome=run_one(target,package,method,settings,method_seed,default_budget)
                    populate_outcome(row,outcome,initial,benchmark,target)
                    if density<1:
                        from prescribed_sip.experiments.diagnostics import sparse_full_diagnostics
                        row.update(sparse_full_diagnostics(target,initial,outcome.support,method))
                    if outcome.support:
                        save_json(root/f'results/designs/{filename}_final.json',support_json(outcome.support))
                    obj=PairObjective(N,target.sip)
                    if outcome.support: obj.initialize(outcome.support)
                    elif method in ('Sinha','DSD-structured'):
                        # Recompute analytic values for scatter/residual artifacts, outside timed run.
                        from prescribed_sip.methods.dsd import projection_moments,projection_factor
                        from prescribed_sip.methods.sinha import sinha_moments,sinha_parameters
                        ps=list(zip(obj.i.tolist(),obj.j.tolist()))
                        if method=='DSD-structured': values=projection_moments(projection_factor(target.pi),ps)
                        else: values=sinha_moments(N,n,sinha_parameters(N,n,target.pi,{p:float(target.sip[p]) for p in ps}))[1]
                        obj.current=np.array([values[p] for p in ps])
                    # Prespecified representative replication only; no selection by accuracy.
                    if replication==0:
                        initial_obj=PairObjective(N,target.sip); initial_obj.initialize(initial)
                        np.savez_compressed(root/f'results/designs/{filename}_pairs.npz',i=obj.i,j=obj.j,
                                            target=obj.target,current=obj.current,initial=initial_obj.current)
                except NotImplementedError as e: row.update({'status':'not_applicable','failure_reason':str(e)})
                except MemoryError as e: row.update({'status':'memory_failure','failure_reason':str(e)})
                except (InvariantError,RuntimeError,ValueError) as e: row.update({'status':'numerical_failure','failure_reason':f'{type(e).__name__}: {e}'})
            if outcome is None: row['runtime_seconds']=perf_counter()-started
            save_json(output,{'row':row,'checkpoints':outcome.checkpoints if outcome else []})
            print(f'[{regime}; workers={execution_workers}] {sid} r{replication+1}/{scenario["replications"]} {method}: {row["status"]} ({row["runtime_seconds"]:.3f}s)',flush=True)
    if replication%10==0:
        print(f'{sid}: replication {replication+1}/{scenario["replications"]}',flush=True)


def finish_execution(config,root,stage=None,artifacts=True):
    collect(root)
    manifest_path=root/'results/manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['status']='configured_stage_execution_finished';manifest['last_requested_stage']=stage or 'all'
    manifest['executed_stages']=sorted(pd.read_csv(root/'results/runs.csv').experiment.unique().tolist())
    save_json(manifest_path,manifest)
    if artifacts:
        from prescribed_sip.experiments.artifacts import generate
        generate(root)


from contextlib import contextmanager


@contextmanager
def execution_lock(root):
    """One scheduler owns the study output; SIGTERM finishes pool shutdown."""
    import fcntl
    import signal
    root=Path(root);(root/'results').mkdir(parents=True,exist_ok=True)
    with (root/'results/execution.lock').open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('study is already running; refusing a duplicate writer') from None
        handle.seek(0);handle.truncate();handle.write(str(os.getpid())+'\n');handle.flush()
        previous=signal.getsignal(signal.SIGTERM)
        def interrupt(signum,frame):raise KeyboardInterrupt('study stop requested')
        signal.signal(signal.SIGTERM,interrupt)
        try:yield
        finally:
            signal.signal(signal.SIGTERM,previous)
            fcntl.flock(handle,fcntl.LOCK_UN)


def execute(config,stage=None,artifacts=True,workers=1,allow_execution_change=False,migration_snapshot=None):
    root=Path(config.get('output','prescribed_sip'))
    with execution_lock(root):
        return _execute(config,stage,artifacts,workers,allow_execution_change,migration_snapshot)


def _execute(config,stage=None,artifacts=True,workers=1,allow_execution_change=False,migration_snapshot=None):
    if not isinstance(workers,int) or workers<1: raise ValueError('workers must be a positive integer')
    root=prepare_execution(config,allow_execution_change,migration_snapshot)
    scenarios=[s for s in expand_scenarios(config) if not stage or s['experiment']==stage]
    if not scenarios: raise ValueError('no configured scenarios for requested stage')
    if workers>1:
        from prescribed_sip.parallel import run_staged
        run_staged(config,root,scenarios,workers,artifacts)
    else:
        for scenario in scenarios:
            for replication in range(scenario['replications']):
                execute_replication(config,root,scenario,replication)
    finish_execution(config,root,stage,artifacts)
    return root


def collect(root):
    rows=[]; checkpoints=[]
    for path in sorted((root/'results/runs').glob('*.json')):
        data=json.loads(path.read_text()); row=data['row']
        row.setdefault('execution_workers',1)
        for seed_key in ['seed','target_seed','initial_seed']:
            if seed_key in row and row[seed_key] is not None: row[seed_key]=str(row[seed_key])
        rows.append(row)
        for point in data.get('checkpoints',[]):
            checkpoints.append({**{k:row[k] for k in ['scenario_id','experiment','N','n','target_type','geometry','method','method_variant','replication','regime','execution_workers']},**point})
    pd.DataFrame(rows).to_csv(root/'results/runs.csv',index=False)
    pd.DataFrame(checkpoints).to_csv(root/'results/checkpoints.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--stage')
    parser.add_argument('--artifacts-only',action='store_true'); parser.add_argument('--no-artifacts',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--resume-execution-change',action='store_true')
    parser.add_argument('--migration-snapshot')
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text())
    if args.artifacts_only:
        from prescribed_sip.experiments.artifacts import generate
        root=Path(config.get('output','prescribed_sip')); collect(root); generate(root)
    else: execute(config,args.stage,not args.no_artifacts,args.workers,args.resume_execution_change,args.migration_snapshot)


if __name__=='__main__': main()
