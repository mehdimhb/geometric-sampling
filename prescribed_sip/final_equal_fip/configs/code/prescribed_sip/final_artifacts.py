"""Independent final-study audit, full-precision tables and publication figures."""
import hashlib
import json
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prescribed_sip.core.dependence import DependenceError
from prescribed_sip.final_study import METHODS, all_tasks, name, task_id, read_support, metrics, utc
from prescribed_sip.run_all import save_json


STATISTICS = ['RMSE', 'RMSE_rho', 'Recovery_rho', 'Runtime', 'Evaluations', 'Support size', 'Entropy']
SHORT = ['Initial', 'Original GFS\n+ restarts', 'GFS-Directed',
         'CP(2)\nenumerated fit', 'DSD\nprojection/order', 'Exact LP/QP']
COLORS = ['#7c8085', '#377eb8', '#1b9e77', '#e69532', '#9363a6', '#c44e52']
EQUAL_TABLE_CAPTION = ('Reconstruction of prescribed second-order inclusion probabilities '
                       'under equal first-order inclusion probabilities.')
UNEQUAL_REFERENCE_CONFIG = Path('/home/bardia/Desktop/GFS_Prescribed_SIP_Final_Results/configs/study.json')


def summarize(runs):
    """SD is between independent repetitions (ddof=1); maxima are worst runs."""
    rows = []
    for (N, n), scenario in runs.groupby(['N', 'n'], sort=True):
        for method in METHODS:
            group = scenario.loc[scenario.Method == method]
            good = group.loc[group['Validation passed'].eq(True)]
            row = {'Method': method, 'N': N, 'n': n, 'Repetitions': len(good),
                   'SIP pairs': N*(N-1)//2, 'Possible samples': comb(int(N), int(n))}
            for metric in STATISTICS:
                values = pd.to_numeric(good[metric], errors='coerce') if metric in good else pd.Series(dtype=float)
                row[metric+' mean'] = values.mean()
                row[metric+' SD'] = values.std(ddof=1) if len(values)>1 else np.nan
                if metric in ['RMSE', 'RMSE_rho']:
                    row[metric+' median'] = values.median()
            for metric in ['Max SIP error', 'FIP max error']:
                row[metric] = good[metric].max() if len(good) else np.nan
            for metric in ['Rho pairs included', 'Rho pairs excluded']:
                row[metric] = group[metric].iloc[0] if len(group) else np.nan
            row['Status'] = '; '.join(sorted(set(group.Status))) if 'Status' in group else ''
            row['Reason'] = '; '.join(sorted(set(group['Stop reason'].dropna().astype(str)))) if 'Stop reason' in group else ''
            rows.append(row)
    return pd.DataFrame(rows)


def publication_table(full):
    def shown(row, metric):
        mean, sd = row[metric+' mean'], row[metric+' SD']
        if pd.isna(mean): return 'NA'
        numeric = (lambda x: f'{x:.2f}') if metric=='Recovery_rho' else (lambda x: f'{x:.3e}')
        return numeric(mean)+(f' ± {numeric(sd)}' if pd.notna(sd) else '')
    rows=[]
    for _, row in full.iterrows():
        result={'Method':row.Method}
        for metric in ['RMSE','RMSE_rho','Recovery_rho']:
            result[metric]=shown(row,metric)
        result['Max SIP error']='NA' if pd.isna(row['Max SIP error']) else f"{row['Max SIP error']:.3e}"
        for metric in ['Runtime','Evaluations','Support size','Entropy']:
            result[metric]=shown(row,metric)
        rows.append(result)
    return pd.DataFrame(rows)


def requested_publication_table(full):
    """Publication layout requested for the equal-FIP counterpart study."""
    columns = ['N', 'n', 'Method', 'RMSE_pi2', 'Recovery (%)',
               'Max SIP error', 'Time (s)', 'Eval.', 'Support']

    def shown(mean, sd, precision=3):
        if pd.isna(mean):
            return 'unavailable'
        value = f'{mean:.{precision}e}' if precision == 3 else f'{mean:.{precision}f}'
        if pd.notna(sd):
            spread = f'{sd:.{precision}e}' if precision == 3 else f'{sd:.{precision}f}'
            value += f' ± {spread}'
        return value

    rows = []
    for (N, n), scenario in full.groupby(['N', 'n'], sort=True):
        initial = scenario.loc[scenario.Method == 'Initial design', 'RMSE mean']
        baseline = float(initial.iloc[0]) if len(initial) else np.nan
        for _, row in scenario.iterrows():
            rmse_mean, rmse_sd = row['RMSE mean'], row['RMSE SD']
            if pd.isna(rmse_mean) or not np.isfinite(baseline) or baseline == 0:
                recovery_mean = recovery_sd = np.nan
            else:
                recovery_mean = 100 * (1 - rmse_mean / baseline)
                recovery_sd = 100 * rmse_sd / baseline if pd.notna(rmse_sd) else np.nan
            rows.append({
                'N': int(N), 'n': int(n), 'Method': row.Method,
                'RMSE_pi2': shown(rmse_mean, rmse_sd),
                'Recovery (%)': shown(recovery_mean, recovery_sd, precision=2),
                'Max SIP error': shown(row['Max SIP error'], np.nan),
                'Time (s)': shown(row['Runtime mean'], row['Runtime SD']),
                'Eval.': shown(row['Evaluations mean'], row['Evaluations SD']),
                'Support': shown(row['Support size mean'], row['Support size SD']),
            })
    return pd.DataFrame(rows, columns=columns)


def export_equal_targets(root, config):
    """Export the already-frozen equal-FIP arrays without regenerating them."""
    if config.get('fip_type', 'unequal') != 'equal':
        raise ValueError('equal target exports require fip_type=equal')
    root = Path(root)
    combined = {}
    for scenario in config['scenarios']:
        scenario_name = name(scenario)
        folder = root / 'targets' / scenario_name
        pi = np.load(folder / 'pi.npy', allow_pickle=False)
        target = np.load(folder / 'Pi_star.npy', allow_pickle=False)
        np.savetxt(root / f'{scenario_name}_fip.csv', pi.reshape(1, -1),
                   delimiter=',', fmt='%.17g')
        np.savetxt(root / f'{scenario_name}_target_sip.csv', target,
                   delimiter=',', fmt='%.17g')
        label = f"N{scenario['N']}"
        combined[f'pi_{label}'] = pi
        combined[f'Pi_star_{label}'] = target
    np.savez(root / 'final_equal_targets.npz', **combined)


def write_comparability_audit(root, equal_config, unequal_config):
    """Compare computational parameters while treating FIP as the intended change."""
    root = Path(root)
    left = {**unequal_config, 'fip_type': unequal_config.get('fip_type', 'unequal')}
    right = {**equal_config, 'fip_type': equal_config.get('fip_type', 'unequal')}
    differences = {}
    for key in sorted(set(left) | set(right)):
        if key == 'notebook':
            continue
        if left.get(key) != right.get(key):
            differences[key] = [left.get(key), right.get(key)]

    notebook_equal = False
    left_notebook, right_notebook = Path(left['notebook']), Path(right['notebook'])
    if left_notebook.is_file() and right_notebook.is_file():
        notebook_equal = hashlib.sha256(left_notebook.read_bytes()).digest() == hashlib.sha256(right_notebook.read_bytes()).digest()
    if not notebook_equal:
        differences['notebook_SHA256'] = [
            hashlib.sha256(left_notebook.read_bytes()).hexdigest() if left_notebook.is_file() else None,
            hashlib.sha256(right_notebook.read_bytes()).hexdigest() if right_notebook.is_file() else None,
        ]

    expected = {'fip_type': ['unequal', 'equal']}
    identical = differences == expected
    lines = [f"Parameters identical to unequal-FIP experiment: {'YES' if identical else 'NO'}", '',
             'Parameters that differ:', 'FIP_TYPE: unequal -> equal',
             'pi vector: unequal values -> constant n/N']
    for scenario in equal_config['scenarios']:
        lines.append(f"  N={scenario['N']}, n={scenario['n']}: pi_i={scenario['n']/scenario['N']:.17g}")
    unexpected = {key: value for key, value in differences.items() if key != 'fip_type'}
    lines += ['', 'Unexpected computational-parameter differences:']
    if unexpected:
        lines += [f'  {key}: {value[0]!r} -> {value[1]!r}' for key, value in unexpected.items()]
    else:
        lines.append('  None.')
    lines += ['', 'The frozen notebook contents are identical.' if notebook_equal else
              'WARNING: the frozen notebook contents differ.']
    (root / 'COMPARABILITY_AUDIT.txt').write_text('\n'.join(lines) + '\n')
    result = {'parameters_identical': identical,
              'parameter_differences': differences,
              'notebook_contents_identical': notebook_equal}
    save_json(root / 'comparability_audit.json', result)
    return result


def save_table(frame, path, caption):
    path=Path(path)
    frame.to_csv(path.with_suffix('.csv'), index=False, na_rep='NA', float_format='%.17g')
    path.with_suffix('.tex').write_text(frame.to_latex(index=False,escape=True,na_rep='NA',
        float_format=lambda x:f'{x:.3e}',caption=caption))


def markdown_table(frame):
    """Render a dependency-free CommonMark table."""
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join('---' for _ in frame.columns) + ' |']
    lines.extend('| ' + ' | '.join(cell(value) for value in row) + ' |'
                 for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def audit(root, config, runs):
    """Re-open serialized probabilities; verify every reported metric independently."""
    root=Path(root); records=[]; tol=config['validation_tolerance']
    tasks=list(all_tasks(config))
    if len(runs)!=len(tasks): raise RuntimeError('Missing or duplicated final repetitions')
    for scenario in config['scenarios']:
        folder=root/'targets'/name(scenario)
        metadata=json.loads((folder/'metadata.json').read_text())
        if metadata['sip_pairs']!=scenario['N']*(scenario['N']-1)//2:
            raise RuntimeError('Scenario label/SIP pair-count mismatch')
        if metadata['sample_count']!=comb(scenario['N'],scenario['n']):
            raise RuntimeError('Incorrect number of fixed-size samples')
        if metadata['accepted_exchanges']!=config['target_exchanges']:
            raise RuntimeError('Target walk did not complete the requested accepted exchanges')
        for filename, digest in metadata['SHA256'].items():
            if hashlib.sha256((folder/filename).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Frozen target changed during reconstruction')
        pi=np.load(folder/'pi.npy'); target=np.load(folder/'Pi_star.npy'); initial=np.load(folder/'initial_Pi.npy')
        # Target probabilities are read only by this postprocessing audit.
        for label,filename,matrix in [('target','target_design_AUDIT_ONLY.json',target),
                                     ('initial','initial_design.json',initial)]:
            design=read_support(folder/filename)
            _, checks, recomputed=metrics(design,pi,target,initial,config)
            delta=float(np.max(abs(matrix-recomputed)))
            if delta>1e-12:raise RuntimeError('Frozen matrix differs from serialized sample probabilities')
            records.append({'Scenario':name(scenario),'Method':label,'Repetition':0,
                            **checks,'Saved SIP discrepancy':delta,'Passed':True})
    for task in tasks:
        scenario=task['scenario']; identifier=task_id(scenario,task['method'],task['repetition'])
        folder=root/'raw_runs'/identifier
        row=json.loads((folder/'result.json').read_text())
        if row['Seed']!=task['seed']:raise RuntimeError('Incorrect reconstruction seed')
        target_folder=root/'targets'/name(scenario)
        pi=np.load(target_folder/'pi.npy'); target=np.load(target_folder/'Pi_star.npy'); initial=np.load(target_folder/'initial_Pi.npy')
        if row['Target SHA256']!=hashlib.sha256(target.tobytes()).hexdigest():raise RuntimeError('A run used a different target')
        if not row['Validation passed']:
            if not row['Stop reason']:raise RuntimeError('Missing skip/failure reason')
            if scenario['N']!=30 or task['method'] not in ['CP(2) — enumerated fit','Exact LP/QP oracle']:
                raise RuntimeError(f"Required final result missing: {identifier}; {row['Stop reason']}")
            continue
        design=read_support(folder/'design.json')
        if design.N!=scenario['N'] or design.n!=scenario['n']:raise RuntimeError('Final population/sample size mismatch')
        values,checks,sip=metrics(design,pi,target,initial,config)
        saved_delta=float(np.max(abs(sip-np.load(folder/'Pi_hat.npy'))))
        if saved_delta>1e-12:raise RuntimeError('Saved final SIP differs from independent recomputation')
        for metric,value in values.items():
            if not np.isclose(row[metric],value,rtol=1e-11,atol=1e-13,equal_nan=True):
                raise RuntimeError(f'Incorrect reported {metric} for {identifier}')
        # Cross-check the rho definition explicitly, independently of reporting helpers.
        i,j=np.triu_indices(scenario['N'],1)
        valid=(pi[i]>config['rho_boundary_tol']) & (pi[i]<1-config['rho_boundary_tol'])
        valid&=(pi[j]>config['rho_boundary_tol']) & (pi[j]<1-config['rho_boundary_tol'])
        if valid.any():
            denominator=np.sqrt(pi[i[valid]]*(1-pi[i[valid]])*pi[j[valid]]*(1-pi[j[valid]]))
            product=pi[i[valid]]*pi[j[valid]]
            rho_star=(target[i[valid],j[valid]]-product)/denominator
            rho_hat=(sip[i[valid],j[valid]]-product)/denominator
            rho_initial=(initial[i[valid],j[valid]]-product)/denominator
            error=np.sqrt(np.mean((rho_hat-rho_star)**2))
            initial_error=np.sqrt(np.mean((rho_initial-rho_star)**2))
            recovery=100*(1-error/initial_error) if initial_error else np.nan
            if not np.isclose(error,row['RMSE_rho'],atol=1e-13,rtol=1e-11,equal_nan=True):
                raise RuntimeError('Explicit rho definition disagrees with reported RMSE_rho')
            if not np.isclose(recovery,row['Recovery_rho'],atol=1e-11,rtol=1e-11,equal_nan=True):
                raise RuntimeError('Recovery_rho formula failed independent audit')
        if abs(checks['probability_sum']-1)>tol:raise RuntimeError('Final probability mass is not one')
        if not checks['positive_samples_correct_size']:raise RuntimeError('Non-fixed-size returned design')
        if checks['max_fixed_size_row_identity_error']>tol:raise RuntimeError('Fixed-size row identity failed')
        budget=(config['directed_evaluations'] if task['method']=='GFS-Directed' else
                config['original_evaluations'] if task['method']=='Original GFS + restarts' else None)
        if budget is not None and row['Evaluations']>budget:raise RuntimeError('Search exceeded candidate budget')
        if row['SIP pairs']!=scenario['N']*(scenario['N']-1)//2:raise RuntimeError('Run SIP pair count is wrong')
        records.append({'Scenario':name(scenario),'Method':task['method'],'Repetition':task['repetition'],
                        **checks,'Saved SIP discrepancy':saved_delta,'Passed':True})
    pd.DataFrame(records).to_csv(root/'validation/validation.csv',index=False,float_format='%.17g')
    save_json(root/'validation/audit.json',{'passed':True,'audited_designs':len(records),
              'completed_runs':int(runs['Validation passed'].sum()),'planned_runs':len(tasks),
              'unavailable_runs':int((~runs['Validation passed']).sum()),'tolerance':tol,'UTC':utc()})
    return records


def save_figure(fig, folder, filename):
    fig.savefig(folder/(filename+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(filename+'.png'),bbox_inches='tight',dpi=300)
    plt.close(fig)


def comparison_plot(scenario, table, folder, metric, logarithmic):
    fig,ax=plt.subplots(figsize=(8,4))
    means=table[metric+' mean'].to_numpy(dtype=float)
    sd=table[metric+' SD'].fillna(0).to_numpy(dtype=float)
    x=np.arange(len(METHODS)); valid=np.isfinite(means)
    # Dots are more readable than bars when the errors span many orders of magnitude.
    for k in x[valid]:
        ax.errorbar(k,means[k],yerr=sd[k],fmt='o',color=COLORS[k],capsize=4,ms=6)
    if logarithmic and np.all(means[valid]>0):ax.set_yscale('log')
    if metric=='Recovery_rho':ax.axhline(100,color='0.65',ls='--',lw=1)
    ax.set_xticks(x,SHORT,fontsize=8)
    ax.set_ylabel({'RMSE':'Raw SIP RMSE','RMSE_rho':r'$\mathrm{RMSE}_{\rho}$',
                   'Recovery_rho':r'$\mathrm{Recovery}_{\rho}$ (%)'}[metric])
    ax.set_title(f"N={scenario['N']}, n={scenario['n']} · mean ± SD across repetitions")
    ax.grid(axis='y',alpha=.2); fig.tight_layout()
    save_figure(fig,folder,metric+'_comparison')


def scenario_plots(root,scenario,runs,full,config):
    folder=root/'plots'/name(scenario);folder.mkdir(parents=True,exist_ok=True)
    target_folder=root/'targets'/name(scenario)
    target=np.load(target_folder/'Pi_star.npy');initial=np.load(target_folder/'initial_Pi.npy')
    title=f"N={scenario['N']}, n={scenario['n']}"
    for metric,logarithmic in [('RMSE_rho',True),('RMSE',True),('Recovery_rho',False)]:
        comparison_plot(scenario,full,folder,metric,logarithmic)
    directed=runs.loc[(runs.Method=='GFS-Directed') & runs['Validation passed']].sort_values('Repetition')
    if len(directed):
        errors=directed.RMSE_rho.to_numpy()
        fig,ax=plt.subplots(figsize=(5,4));ax.boxplot(errors,positions=[1],widths=.3,showfliers=False)
        jitter=np.random.default_rng(0).uniform(-.1,.1,len(errors))
        ax.scatter(1+jitter,errors,s=20,c=COLORS[2],alpha=.75)
        ax.set_xticks([1],[f'GFS-Directed ({len(errors)} independent runs)'])
        ax.set_ylabel(r'$\mathrm{RMSE}_{\rho}$')
        if np.all(errors>0):ax.set_yscale('log')
        ax.set_title(title+' · repetition distribution');ax.grid(axis='y',alpha=.2);fig.tight_layout()
        save_figure(fig,folder,'Directed_RMSE_rho_distribution')
        fig,ax=plt.subplots(figsize=(7,4)); grid=np.unique(np.r_[0,np.geomspace(1,config['directed_evaluations'],250).astype(int)])
        trajectories=[]
        for _,row in directed.iterrows():
            path=root/'raw_runs'/task_id(scenario,'GFS-Directed',int(row.Repetition))/'convergence.csv'
            trace=pd.read_csv(path).sort_values('evaluations').drop_duplicates('evaluations',keep='last')
            ax.plot(trace.evaluations,trace.RMSE_rho,color=COLORS[2],alpha=.13,lw=.8)
            # Connect sampled checkpoints; coarse recording gaps are not algorithm stalls.
            values=trace.RMSE_rho.to_numpy()
            if np.all(values>0):
                interpolated=np.exp(np.interp(grid,trace.evaluations,np.log(values)))
            else:
                interpolated=np.interp(grid,trace.evaluations,values)
            trajectories.append(interpolated)
        trajectories=np.asarray(trajectories)
        median=np.median(trajectories,axis=0);mean=np.mean(trajectories,axis=0)
        ax.plot(grid,median,color='#06634f',lw=2,label='Median checkpoint trajectory')
        ax.plot(grid,mean,color='#163d77',lw=1.5,ls='--',label='Mean checkpoint trajectory')
        if np.all(trajectories>0):ax.set_yscale('log')
        ax.set_xlabel('Scored candidate evaluations');ax.set_ylabel(r'$\mathrm{RMSE}_{\rho}$')
        ax.set_title(title+f' · GFS-Directed ({len(directed)} runs)');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
        save_figure(fig,folder,'Directed_convergence_RMSE_rho')
        pd.DataFrame({'evaluations':grid,'median_RMSE_rho':median,'mean_RMSE_rho':mean,
                      'SD_RMSE_rho':np.std(trajectories,axis=0,ddof=1) if len(directed)>1 else np.nan
                      }).to_csv(folder/'Directed_convergence_aggregate.csv',index=False,float_format='%.17g')
        # Best by primary scale-free reporting metric; raw RMSE remains search objective.
        best=directed.loc[directed.RMSE_rho.idxmin()]
        reconstructed=np.load(root/'raw_runs'/task_id(scenario,'GFS-Directed',int(best.Repetition))/'Pi_hat.npy')
        fig,axes=plt.subplots(1,3,figsize=(12,3.8));limit=max(target.max(),reconstructed.max())
        error=reconstructed-target;err_limit=max(abs(error).max(),1e-18)
        for ax,matrix,label,cmap,bounds in zip(axes,[target,reconstructed,error],
              ['Target SIP',f'Directed SIP · repetition {int(best.Repetition)}','Reconstruction − target'],
              ['viridis','viridis','RdBu_r'],[(0,limit),(0,limit),(-err_limit,err_limit)]):
            im=ax.imshow(matrix,cmap=cmap,vmin=bounds[0],vmax=bounds[1]);fig.colorbar(im,ax=ax,shrink=.8)
            ax.set_title(label,fontsize=10);ax.set_xlabel('Original unit ID');ax.set_ylabel('Original unit ID')
        fig.suptitle(title+' · best Directed repetition by RMSEρ');fig.tight_layout()
        save_figure(fig,folder,'Directed_SIP_heatmaps')
        pairs=np.triu_indices(scenario['N'],1)
        fig,ax=plt.subplots(figsize=(4.8,4.5));ax.scatter(target[pairs],reconstructed[pairs],s=13,alpha=.7,color=COLORS[2])
        top=max(target[pairs].max(),reconstructed[pairs].max());ax.plot([0,top],[0,top],color='0.3',lw=1,ls='--')
        ax.set_xlabel('Target πᵢⱼ');ax.set_ylabel('Reconstructed πᵢⱼ');ax.set_title(title+' · Directed SIP comparison')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);fig.tight_layout();save_figure(fig,folder,'Directed_SIP_scatter')
        save_json(folder/'representative_Directed.json',{'selection':'minimum independently recomputed RMSE_rho',
                   'Repetition':int(best.Repetition),'Seed':int(best.Seed),'RMSE':best.RMSE,'RMSE_rho':best.RMSE_rho})
    difference=target-initial;limit=max(abs(difference).max(),1e-18)
    fig,ax=plt.subplots(figsize=(5,4.5));im=ax.imshow(difference,cmap='RdBu_r',vmin=-limit,vmax=limit)
    fig.colorbar(im,ax=ax,label='Target − original-order initial SIP')
    ax.set_title(title+' · frozen target discrepancy');ax.set_xlabel('Original unit ID');ax.set_ylabel('Original unit ID')
    fig.tight_layout();save_figure(fig,folder,'Target_initial_SIP_discrepancy')


def generate_artifacts(root,config):
    root=Path(root)
    runs=pd.DataFrame([json.loads((root/'raw_runs'/task_id(t['scenario'],t['method'],t['repetition'])/'result.json').read_text())
                       for t in all_tasks(config)])
    runs.to_csv(root/'raw_runs/all_runs.csv',index=False,float_format='%.17g')
    audit(root,config,runs)
    full=summarize(runs)
    save_table(full,root/'tables/Table_All_Scenarios','All frozen-target prescribed-SIP scenarios. SDs are sample SDs across independent reconstruction repetitions; maxima are worst-case over repetitions.')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    combined=[]
    for k,scenario in enumerate(config['scenarios'],1):
        table=full.loc[(full.N==scenario['N']) & (full.n==scenario['n'])].reset_index(drop=True)
        scenario_runs=runs.loc[(runs.N==scenario['N']) & (runs.n==scenario['n'])]
        basename=f'Table_{k}_{name(scenario)}'
        save_table(table,root/'tables'/basename,f"Prescribed-SIP reconstruction, N={scenario['N']}, n={scenario['n']}. One frozen target; independent reconstruction repetitions.")
        compact=publication_table(table)
        save_table(compact,root/'tables'/(basename+'_Compact'),f"N={scenario['N']}, n={scenario['n']}. Means ± sample SD; single-run SD omitted. Max SIP error is the maximum across runs.")
        combined.append(compact.assign(Scenario=name(scenario)))
        scenario_plots(root,scenario,scenario_runs,table,config)
    save_table(pd.concat(combined,ignore_index=True),root/'tables/Table_All_Scenarios_Compact','Compact principal results. Means ± sample SD; maxima across runs.')
    requested = None
    if config.get('fip_type', 'unequal') == 'equal':
        requested = requested_publication_table(full)
        save_table(requested, root/'tables/Equal_FIP_Results', EQUAL_TABLE_CAPTION)
        (root/'tables/Equal_FIP_Results.md').write_text(
            EQUAL_TABLE_CAPTION + '\n\n' + markdown_table(requested) + '\n')
        export_equal_targets(root, config)
        if UNEQUAL_REFERENCE_CONFIG.is_file():
            write_comparability_audit(
                root, config, json.loads(UNEQUAL_REFERENCE_CONFIG.read_text()))
        else:
            (root/'COMPARABILITY_AUDIT.txt').write_text(
                'Parameters identical to unequal-FIP experiment: NO\n\n'
                f'Authoritative unequal configuration not found: {UNEQUAL_REFERENCE_CONFIG}\n')
    env=json.loads((root/'configs/environment.json').read_text())
    unavailable=runs.loc[~runs['Validation passed']]
    limitations=[]
    for _,row in unavailable.iterrows():limitations.append(f"{row.Scenario}, {row.Method}: {row['Status']}; {row['Stop reason']}")
    exceptional=runs.loc[runs.Status.isin(['timeout','numerical_failure','calibration_limit','uncertified_qp'])]
    for _,row in exceptional.iterrows():
        if row['Validation passed']:
            limitations.append(f"{row.Scenario}, {row.Method}, repetition {row.Repetition}: returned valid design; status {row.Status}; {row['Stop reason']}")
    scenario_text='\n'.join(f"  N={s['N']}, n={s['n']}, pairs={s['N']*(s['N']-1)//2}, possible samples={comb(s['N'],s['n'])}; population seed={s['population_seed']}, permutation seed={s['target_seed']}, exchange seed={s['target_seed']+1}" for s in config['scenarios'])
    fip_description = ('equal pi_i = n/N.' if config.get('fip_type', 'unequal') == 'equal' else
                       'unequal pi_i using x_i ~ Uniform(0.65,1.35), pi_i = n*x_i/sum(x), as in the working notebook.')
    readme=f"""FINAL PRESCRIBED-SIP STUDY {'SMOKE CHECK ONLY' if config['smoke'] else 'FINAL EXECUTED BUDGETS'}
Completed UTC: {utc()}
Git commit: {env['git_commit']} (working source snapshot includes the user's uncommitted package changes).
Python/dependency versions and source hashes: configs/environment.json.
Source snapshot and working notebook: configs/code/. Full settings: configs/study.json.

Scenarios: {fip_description} Units are ordered on a line.
{scenario_text}

ONE frozen reachable target per scenario: reproducibly permute population order while preserving original IDs/pi_i; construct the package's standard fixed-size GFS design on that order; perform exactly {config['target_exchanges']} accepted non-null random package Design.exchange operations (M=1, alpha=0.75); take endpoint SIP and restore original unit order. The objective is not used to select the target endpoint.
targets/*/metadata.json documents seeds, permutation, first-order probabilities, initial discrepancy and checksums. Target probability distributions are saved as AUDIT_ONLY and never loaded by reconstruction workers.
Both GFS reconstruction methods begin at the same standard original-order fixed-size GFS design p0. Competitors retain their existing native initializations: CP(2)'s dual starts at zero parameters after uniform-family FIP calibration, DSD first tries the original projection order then random orders, and the LP/QP oracle solves the sparse constraints with its existing solver. All methods receive the prescribed FIP and frozen SIP target; p0 is the COMMON recovery baseline. No reconstruction receives the target probability distribution.

Methods:
  Initial design: one deterministic reference per scenario.
  Original GFS + restarts: current actual package GreedyBestFirstSearch, frontier=50, top_k=1, new_orders=2, new_exchanges=10, exchange coefficient=0.75, partitions=1; documented wrapper restarts from best design after frontier exhaustion with 3 random package exchanges. No new optimizer replaces the package search. {config['original_repetitions']} independent repetitions, at most {config['original_evaluations']} scored evaluations. N=30 per-run time cap {config['original_N30_seconds_per_run']} s. The per-run details/restarts.csv give actual restart counts and settings. Stagnation is disabled as in the active working notebook.
  GFS-Directed: unchanged residual-directed four-support mass exchanges, exact feasible optimal step along each direction, best candidate from batch {config['directed_batch']}; {config['directed_repetitions']} independent seeds, at most {config['directed_evaluations']} scored evaluations each. Raw SIP objective unchanged; early stop raw SIP RMSE <= {config['raw_rmse_stop']:g}. Inclusive boundary is an explicitly requested stopping adaptation only.
  CP(2) — enumerated fit: existing full-support log-partition exponential-family fit, L-BFGS gtol=1e-11 / ftol=1e-15, with mandatory final Newton FIP calibration. Up to {config['cp2_evaluations']} dual evaluations and {config['cp2_seconds']} s for fitting, followed by separately reported mandatory finalization. Enumeration threshold {config['enumeration_limit']}; includes a genuine N=30 attempt with 142506 samples. Limitations of this implementation do not imply a theoretical scalability limit of CP(2).
  DSD — projection/order fit: existing projection determinant design, {config['dsd_trials']} order fitting trials, full sample enumeration for independent returned probabilities.
  Exact LP/QP oracle: existing sparse LP feasibility / QP oracle; {config['exact_seconds']} s LP solver budget. N=30 optional attempt is bounded by the same solver budget. A certified feasible LP solution suffices for these reachable targets.
All algorithms use float64. Runs use {config['workers']} independent processes with BLAS threads restricted to one. Runtime is method wall time under parallel load; probability/SIP serialization and audit times are outside algorithm runtime. Evaluations count GFS scored candidates, CP(2) dual evaluations, DSD trials; exact LP/QP evaluations are 0 (solver iterations are separate). The auxiliary Task wall seconds field includes setup and method execution, measured before the final independent audit/serialization. Peak process RSS is a process-lifetime high-water mark; workers are reused.

Metrics: raw RMSE over ALL i<j SIP pairs. RMSE_rho uses the SAME prescribed pi_i denominators sqrt(pi_i(1-pi_i)pi_j(1-pi_j)); pairs involving pi within {config['rho_boundary_tol']:g} of 0 or 1 are excluded and counts explicitly saved. Recovery_rho=100*(1-final_RMSE_rho/initial_RMSE_rho). Undefined recovery is NA. Directed optimization minimizes raw SIP error, not RMSE_rho.
Tables: full precision CSV; between-run sample SD (ddof=1), median raw/rho RMSE, single-run SD=NA. SD measures reconstruction randomness conditional on the one fixed population/target in each case. Max SIP error/FIP error are maxima over repetitions. Compact tables show mean ± SD. Each scenario has PDF and 300-dpi PNG figures. Directed convergence connects the existing log-spaced recorded incumbent checkpoints. Mean/median aggregate curves use interpolation linear in log(RMSE_rho) between checkpoints, with stopped endpoints held constant to the common candidate budget; aggregate CSV values between recorded points are interpolated. Raw per-run convergence.csv retains all actual recorded values. Best Directed heatmap/scatter selection uses final independently recomputed RMSE_rho and is separate from across-run mean results.
Independent audit: validation/validation.csv and audit.json, reconstructed from serialized final probabilities; mass, FIP, full SIP, fixed sample sizes, fixed-size row identities, support bound, pair counts, frozen-target hashes, reported metrics and recovery are checked with tolerance {config['validation_tolerance']:g} (saved SIP agreement 1e-12).

Reproduction from the project root (same environment):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m prescribed_sip.final_study --fip-type {config.get('fip_type', 'unequal')} --output NEW_OUTPUT_FOLDER
To resume an interrupted study, run ./resume.sh in this folder. It imports the frozen source snapshot and skips completed repetitions. Existing completed run and target files are retained and checked; only unfinished repetitions restart. Local computation does not require chat/API calls or tokens. configs/reconstruction_seeds.json records EVERY exact reconstruction seed. Code snapshot preserves this runner and the algorithms.

Unavailable methods / computation limitations:
"""
    readme+='\n'.join(limitations) if limitations else 'None. All planned methods returned independently validated probability designs.'
    readme+='\n\nPrincipal results: tables/Table_All_Scenarios_Compact.csv\n'
    if requested is not None:
        readme+='Requested equal-FIP publication table: tables/Equal_FIP_Results.csv\n'
    (root/'README.txt').write_text(readme)
    save_json(root/'validation/limitations.json',{'unavailable':unavailable.to_dict('records'),
              'returned_design_computation_limits':exceptional.to_dict('records')})
    if requested is not None:
        print('\n'+EQUAL_TABLE_CAPTION)
        print(requested.to_string(index=False), flush=True)
