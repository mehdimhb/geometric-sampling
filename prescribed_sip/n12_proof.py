"""Bounded two-fixed-target N12,n4 proof, with an enforced runtime approval gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
from time import perf_counter
import numpy as np
import pandas as pd
from prescribed_sip.core.design import Support
from prescribed_sip.core.objective import PairObjective
from prescribed_sip.experiments.n12_targets import MASTER,PAIRS,build_case,make_target,independent_sip,validate_design
from prescribed_sip.experiments.targets import seed_for
from prescribed_sip.methods.n12_original import run_original
from prescribed_sip.methods.exact import exact_benchmark,space_for
from prescribed_sip.methods.cp2 import fit_cp2
from prescribed_sip.methods.sinha import sinha_parameters,sinha_support
from prescribed_sip.methods.dsd import projection_factor,projection_moments
from prescribed_sip.run_all import save_json,support_json,utc_now,execution_lock

ROOT=Path('prescribed_sip/results/N12_n4')


def read_support(path):
    data=json.loads(Path(path).read_text())
    return Support(data['N'],data['n'],dict(zip(map(tuple,data['samples']),data['probabilities'])))


def machine_info():
    cpus=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1))
    core_ids=set()
    for cpu in cpus:
        base=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        if (base/'core_id').exists():core_ids.add(((base/'physical_package_id').read_text().strip(),(base/'core_id').read_text().strip()))
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    logical=len(cpus);workers=max(1,min(int(.75*logical),logical-2))
    return {'logical_CPUs_system':os.cpu_count(),'logical_CPUs_available':logical,'physical_cores_available':len(core_ids) or None,
            'CPU_affinity':cpus,'workers':workers,'free_logical_CPU_slots':logical-workers,
            'RAM_total_bytes':mem['MemTotal'],'RAM_available_bytes':mem['MemAvailable'],'detected_UTC':utc_now(),
            'library_threads_per_worker':1,
            'numerical_thread_environment':{key:os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
            'runtime_comparison':'runtimes measured under documented concurrent independent-run load; not isolated hardware benchmarks'}


def case_folder(root,case):return Path(root)/('target_equal' if case=='E' else 'target_unequal')


def prepare(root):
    root=Path(root)
    for folder in ['raw_runs','designs','logs','reports','figures','tables','pilot','source_snapshots']:(root/folder).mkdir(parents=True,exist_ok=True)
    machine=machine_info();save_json(root/'machine.json',machine)
    print(f'Hardware: {machine["logical_CPUs_available"]} logical CPUs, {machine["physical_cores_available"]} physical cores, {machine["RAM_available_bytes"]/2**30:.2f} GiB available RAM; workers={machine["workers"]}; numerical threads=1.',flush=True)
    for case in ['E','U']:
        folder=case_folder(root,case);folder.mkdir(exist_ok=True)
        if (folder/'metadata.json').exists():continue
        pi,x,initial=build_case(case);walk_seed=seed_for(MASTER,'N12proof',case,'target_walk')
        target,trajectory,metadata=make_target(initial,pi,walk_seed)
        metadata.update({'case':case,'N':12,'n':4,'pi':pi.tolist(),'x':x.tolist(),'target_seed':walk_seed,
                         'initial_seed':MASTER,'coordinates':'equally spaced ordered line, units 0..11'})
        save_json(folder/'p0.json',support_json(initial));save_json(folder/'pstar.json',support_json(target))
        np.save(folder/'Pi0.npy',independent_sip(initial));np.save(folder/'Pistar.npy',independent_sip(target))
        np.save(folder/'pi.npy',pi);np.save(folder/'x.npy',x)
        pd.DataFrame(trajectory).to_csv(folder/'trajectory.csv',index=False)
        save_json(folder/'metadata.json',metadata)
        print(f'Target {case}: snapshot={metadata["selected_exchanges"]} exchanges of {metadata["walk_exchanges"]}; H {metadata["H0"]:.6f}->{metadata["Hstar"]:.6f}; support {metadata["support0"]}->{metadata["supportstar"]}; RMSE from initial={metadata["initial_RMSE"]:.6g}; d_rho={metadata["d_rho"]:.6g}; maximum FIP error={metadata["validation"]["max_FIP_error"]:.3g}.',flush=True)
    return machine


def summarize_support(support,pi,target,initial_rmse):
    sip=independent_sip(support);error=sip[PAIRS]-target[PAIRS];loss=float(error@error);rmse=float(np.sqrt(loss/66))
    return {'final_D2':loss,'final_RMSE':rmse,'final_Dinf':float(np.max(abs(error))),
            'recovery_percent':float(100*(1-rmse/initial_rmse)),'validation':validate_design(support,pi,sip)},sip


def fit_dsd(pi,target,seed,evaluations=5000):
    """Fit only the existing projection construction's unit-order permutation."""
    rng=np.random.default_rng(seed);started=perf_counter();best=None;best_loss=np.inf;trace=[];last_improvement=0
    def factor(order):
        V=np.empty((12,4));V[order]=projection_factor(pi[order]);return V
    order=np.arange(12);V=factor(order)
    base_sip=np.asarray([v for v in projection_moments(V,list(zip(*PAIRS))).values()])
    baseline=float(np.sqrt(np.mean((base_sip-target[PAIRS])**2)))
    for count in range(1,evaluations+1):
        if count==1:proposal=order.copy()
        elif count%100==0:proposal=rng.permutation(12)
        else:
            proposal=order.copy();i,j=rng.choice(12,2,replace=False);proposal[i],proposal[j]=proposal[j],proposal[i]
        trial=factor(proposal);K=trial@trial.T;values=(np.outer(pi,pi)-K*K)[PAIRS]
        error=values-target[PAIRS];loss=float(error@error)
        if loss<best_loss:
            best_loss=loss;order=proposal;best=trial;last_improvement=count
        if count==1 or count%50==0:
            bestK=best@best.T;residual=(np.outer(pi,pi)-bestK*bestK)[PAIRS]-target[PAIRS]
            trace.append({'candidates':count,'D2':best_loss,'RMSE':float(np.sqrt(best_loss/66)),
                          'Dinf':float(np.max(abs(residual))),'elapsed_seconds':perf_counter()-started})
    mass={}
    for sample in space_for(12,4).samples:
        p=float(np.linalg.det(best[list(sample)])**2)
        if p>0:mass[sample]=p
    support=Support(12,4,mass);validation=validate_design(support,pi)
    analytic=np.outer(pi,pi)-(best@best.T)**2;np.fill_diagonal(analytic,pi)
    validation['max_SIP_recompute_error']=float(np.max(abs(independent_sip(support)-analytic)))
    if validation['max_SIP_recompute_error']>1e-10:raise RuntimeError('DSD determinant probabilities disagree with kernel moments')
    return support,{'runtime_seconds':perf_counter()-started,'candidate_evaluations':evaluations,'trace':trace,
        'details':{'construction':'existing projection_factor, with fitted unit ordering only','best_order':order.tolist(),
                   'order_search':'uniform random pair swaps of incumbent; independent random order every 100 candidates',
                   'initial_fixed_order_RMSE':baseline,'kernel_family':'restricted ordered rotation projections; not arbitrary prescribed SIP',
                   'last_improvement_candidate':last_improvement,'seed':int(seed),'validation':validation}}


def save_row(root,case,method,replication,row,support,pi,target,initial_rmse):
    root=Path(root);slug=method.replace(' ','_').replace('/','_').replace('(','').replace(')','')
    stem=f'{case}_{slug}_r{replication:02d}'
    row.update({'case':case,'method':method,'replication':replication})
    if support is not None:
        metrics,sip=summarize_support(support,pi,target,initial_rmse)
        prior_validation=row.get('validation',{});row.update(metrics);row['validation'].update(prior_validation)
        row['design_path']=f'designs/{stem}.json';row['sip_path']=f'designs/{stem}_SIP.npy'
        save_json(root/row['design_path'],support_json(support));np.save(root/row['sip_path'],sip)
        for entry in row.get('trace',[]):entry['recovery_percent']=float(100*(1-entry['RMSE']/initial_rmse))
    row.setdefault('trace',[]);row.setdefault('seed',None);row.setdefault('candidate_evaluations',0)
    row.setdefault('runtime_seconds',0.);row.setdefault('validation',{});row.setdefault('details',{})
    save_json(root/f'raw_runs/{stem}.json',row)
    if row['trace']:pd.DataFrame(row['trace']).to_csv(root/f'raw_runs/{stem}_convergence.csv',index=False)
    return row


def stochastic_job(root,case,method,replication,seed):
    root=Path(root);folder=case_folder(root,case);pi=np.load(folder/'pi.npy');target=np.load(folder/'Pistar.npy')
    metadata=json.loads((folder/'metadata.json').read_text());initial_rmse=metadata['initial_RMSE']
    slug=method.replace(' ','_').replace('/','_').replace('(','').replace(')','')
    existing=root/f'raw_runs/{case}_{slug}_r{replication:02d}.json'
    if existing.exists():return json.loads(existing.read_text())
    if method=='DSD':
        support,row=fit_dsd(pi,target,seed)
        row.update({'status':'completed','seed':int(seed)})
        row['validation']=row['details']['validation']
    else:row,support=run_original(pi,target,seed,high_precision=method.endswith('high precision'))
    row['worker_PID']=os.getpid();row['peak_process_RSS_MB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    return save_row(root,case,method,replication,row,support,pi,target,initial_rmse)


def deterministic_runs(root):
    root=Path(root)
    for case in ['E','U']:
        folder=case_folder(root,case);pi=np.load(folder/'pi.npy');target=np.load(folder/'Pistar.npy');initial=read_support(folder/'p0.json')
        initial_rmse=json.loads((folder/'metadata.json').read_text())['initial_RMSE']
        if not (root/f'raw_runs/{case}_Initial_r00.json').exists():
            save_row(root,case,'Initial',0,{'status':'initial','seed':MASTER},initial,pi,target,initial_rmse)
        if not (root/f'raw_runs/{case}_Exact_LP_QP_r00.json').exists():
            benchmark=exact_benchmark(12,4,pi,PairObjective(12,target),seconds=60)
            support=benchmark.pop('support',None)
            if benchmark['status']!='certified_feasible':raise RuntimeError(f'known feasible target {case} not certified by LP: {benchmark}')
            save_row(root,case,'Exact LP/QP',0,{'status':benchmark['status'],'runtime_seconds':benchmark['runtime_seconds'],
                'candidate_evaluations':0,'details':benchmark},support,pi,target,initial_rmse)
        if not (root/f'raw_runs/{case}_CP2_r00.json').exists():
            outcome=fit_cp2(12,4,pi,PairObjective(12,target),5000,180,initial=initial)
            row={'status':outcome.status,'failure_reason':outcome.failure_reason,'runtime_seconds':outcome.runtime_seconds,
                 'candidate_evaluations':outcome.candidate_moves,'validation':outcome.validation,'details':outcome.details}
            save_row(root,case,'CP(2)',0,row,outcome.support,pi,target,initial_rmse)
        if not (root/f'raw_runs/{case}_Sinha_r00.json').exists():
            started=perf_counter();b=sinha_parameters(12,4,pi,{pair:float(target[pair]) for pair in zip(*PAIRS)})
            try:support=sinha_support(12,4,b);row={'status':'completed','details':{'minimum_mixture_coefficient':min(b.values())}}
            except ValueError as error:
                support=None;row={'status':'NA — theoretical restrictions not satisfied','failure_reason':str(error),
                                  'details':{'minimum_mixture_coefficient':min(b.values())},
                                  'final_RMSE':None,'final_D2':None,'final_Dinf':None,'recovery_percent':None}
            row['runtime_seconds']=perf_counter()-started
            save_row(root,case,'Sinha',0,row,support,pi,target,initial_rmse)


def benchmark(root,machine,candidates=500):
    root=Path(root);rows=[]
    for case in ['E','U']:
        folder=case_folder(root,case);pi=np.load(folder/'pi.npy');target=np.load(folder/'Pistar.npy')
        row,support=run_original(pi,target,seed_for(MASTER,'N12proof',case,'pilot'),evaluations=candidates)
        row['case']=case;save_json(root/f'pilot/{case}_Original_GFS.json',row);rows.append(row)
        print(f'Pilot {case}: {row["candidate_evaluations"]} candidates in {row["runtime_seconds"]:.3f}s; {row["details"]["evaluations_per_second"]:.2f} candidates/s; criterion {row["details"]["average_candidate_criterion_seconds"]*1000:.3f} ms/candidate; peak process RSS={row["details"]["peak_process_RSS_MB"]:.1f} MiB; profile={row["details"]["profile"]}.',flush=True)
    # Protect against growing support/precision variants and sibling contention.
    per_run=max(row['runtime_seconds']/max(1,row['candidate_evaluations'])*5000 for row in rows)*3
    # 80 paired GFS runs, plus 40 cheaper projection-order fits; conservative 20s/DSD.
    cpu_total=80*per_run+40*20
    estimated=cpu_total/machine['workers']*2+420
    report={'pilot_candidates':candidates,'original_pilots':rows,'projected_seconds_per_full_GFS_run':per_run,
        'projected_serial_seconds_for_20_GFS_repetitions':20*per_run,
        'projected_parallel_seconds_for_20_GFS_repetitions':20*per_run/machine['workers']*2,
        'estimated_total_seconds':estimated,'workers':machine['workers'],
        'estimate_assumptions':'3x serial pilot extrapolation for support growth/precision; 2x parallel slowdown; 40 DSD fits at 20s each; 420s oracle/calibration/artifact allowance',
        'full_GFS_runs':80,'DSD_runs':40,'repetitions_per_case_method':20,
        'requires_approval':estimated>7200,'reported_UTC':utc_now()}
    save_json(root/'benchmark.json',report)
    print(f'BEFORE FULL EXECUTION: projected GFS repetition <=approximately {per_run:.1f}s; 20 repetitions serial {20*per_run/60:.1f}min; complete experiment conservative estimate {estimated/60:.1f}min with {machine["workers"]} workers. Approval gate: {"STOP (>2 hours)" if estimated>7200 else "proceed (<2 hours)"}.',flush=True)
    return report


def run(root=ROOT,prepare_only=False):
    root=Path(root)
    with execution_lock(root):
        machine=prepare(root)
        estimates=benchmark(root,machine)
        if prepare_only or estimates['requires_approval']:
            save_json(root/'progress.json',{'status':'awaiting_execution' if prepare_only else 'approval_required',
                      'estimated_total_seconds':estimates['estimated_total_seconds'],'updated_UTC':utc_now()})
            return
        source_files=list(Path('graphical_sampling').rglob('*.py'))+list(Path('prescribed_sip').rglob('*.py'))
        fingerprints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
        for source in source_files:
            snapshot=root/'source_snapshots'/(str(source)+'.txt')
            snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(source.read_bytes())
        manifest={'status':'running','N':12,'n':4,'cases':['E','U'],'master_seed':MASTER,'repetitions':20,'scored_candidate_budget':5000,
                  'workers':machine['workers'],'source_SHA256':fingerprints,'started_UTC':utc_now(),
                  'target_SHA256':{case:hashlib.sha256((case_folder(root,case)/'Pistar.npy').read_bytes()).hexdigest() for case in ['E','U']},
                  'runtime_gate_estimate_seconds':estimates['estimated_total_seconds'],'score':'D2','parallel_start_method':'spawn',
                  'scope':'exactly two fixed targets; no larger population or broad-study restart'}
        save_json(root/'manifest.json',manifest)
        jobs=[]
        for case in ['E','U']:
            for method in ['Original GFS','Original GFS high precision','DSD']:
                for rep in range(20):
                    # Paired common random streams isolate score-precision effects.
                    tag='GFS' if method.startswith('Original GFS') else 'DSD'
                    seed=seed_for(MASTER,'N12proof',case,tag,rep)
                    jobs.append((case,method,rep,seed))
        pd.DataFrame(jobs,columns=['case','method','replication','seed']).to_csv(root/'seeds.csv',index=False)
        deterministic_runs(root)
        finished=0;started=perf_counter()
        with ProcessPoolExecutor(max_workers=machine['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(stochastic_job,str(root),*job):job for job in jobs}
            for future in as_completed(futures):
                row=future.result();finished+=1
                progress={'status':'running','completed_repetitions':finished,'total_repetitions':len(jobs),
                          'remaining_repetitions':len(jobs)-finished,'workers':machine['workers'],'elapsed_seconds':perf_counter()-started,'updated_UTC':utc_now(),
                          'last_case':row['case'],'last_method':row['method'],'last_RMSE':row['final_RMSE']}
                save_json(root/'progress.json',progress)
                print(f'[{finished}/{len(jobs)}] {row["case"]} {row["method"]} rep{row["replication"]+1}: RMSE={row["final_RMSE"]:.6g}; recovery={row["recovery_percent"]:.2f}%; time={row["runtime_seconds"]:.2f}s; remaining={len(jobs)-finished}.',flush=True)
        from prescribed_sip.experiments.n12_artifacts import generate
        generate(root)
        manifest.update({'status':'completed','finished_UTC':utc_now(),'execution_seconds':perf_counter()-started,
                         'results_rows':len(list((root/'raw_runs').glob('*.json')))})
        save_json(root/'manifest.json',manifest)
        progress.update({'status':'completed','remaining_repetitions':0,'updated_UTC':utc_now()});save_json(root/'progress.json',progress)
        print('Completed exactly the two N12,n4 problems; stopping. No larger experiments launched.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--output',default=str(ROOT))
    args=parser.parse_args();run(Path(args.output),args.prepare_only)
