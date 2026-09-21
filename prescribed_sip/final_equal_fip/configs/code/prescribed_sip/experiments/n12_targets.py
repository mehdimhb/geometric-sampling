"""Two frozen reachable targets from objective-blind mass-exchange walks."""
from itertools import combinations
import numpy as np
from prescribed_sip.methods.package_gfs import initial_design,extract_support,seeded_package
from prescribed_sip.methods.exchange_search import random_candidate

MASTER=20260918
N,n=12,4
PAIRS=np.triu_indices(N,1)


def independent_sip(support):
    matrix=np.zeros((support.N,support.N),dtype=np.float64)
    for sample,probability in support.mass.items():
        for i in sample:
            matrix[i,i]+=probability
        for i,j in combinations(sample,2):
            matrix[i,j]+=probability;matrix[j,i]+=probability
    return matrix


def build_case(case):
    if case=='E':
        x=np.ones(N);pi=np.full(N,n/N,dtype=np.float64)
    elif case=='U':
        x=np.random.default_rng(MASTER+41).uniform(.65,1.35,N)
        pi=n*x/x.sum()
    else:raise ValueError('case must be E or U')
    if np.min(pi)<=.1 or np.max(pi)>=.6:raise ValueError('unequal vector outside moderate range')
    coords=np.column_stack([np.linspace(0,1,N),np.zeros(N)])
    with seeded_package(MASTER):initial=extract_support(initial_design(pi,coords))
    initial.validate(pi)
    return pi,x,initial


def validate_design(support,pi,saved_sip=None):
    vals=support.validate(pi)
    if min(support.mass.values()) < -1e-12:raise RuntimeError('returned negative probability')
    sip=independent_sip(support)
    vals.update({'max_SIP_recompute_error':0. if saved_sip is None else float(np.max(abs(sip-saved_sip))),
        'max_fixed_size_row_identity_error':float(np.max(abs(sip.sum(axis=1)-np.diag(sip)-(support.n-1)*pi))),
        'total_pair_identity_error':abs(float(np.triu(sip,1).sum())-support.n*(support.n-1)/2),
        'max_SIP_diagonal_FIP_error':float(np.max(abs(np.diag(sip)-pi))),
        'positive_samples_correct_size':all(len(s)==support.n for s,p in support.mass.items() if p>0)})
    if max(vals['max_SIP_recompute_error'],vals['max_fixed_size_row_identity_error'],vals['total_pair_identity_error'])>1e-10:
        raise RuntimeError('independent SIP validation failed')
    return vals


def make_target(initial,pi,seed,exchanges=5000,interval=50):
    rng=np.random.default_rng(seed);working=initial.copy();original=independent_sip(initial)
    denominator=np.sqrt(np.outer(pi*(1-pi),pi*(1-pi)))[PAIRS]
    trajectory=[];snapshots=[];accepted=0;attempts=0
    def snapshot():
        current=independent_sip(working);delta=(current-original)[PAIRS]
        row={'exchanges':accepted,'entropy':working.entropy(),'support':len(working.mass),
             'effective_support':float(np.exp(working.entropy())),
             'RMSE_from_initial':float(np.sqrt(np.mean(delta*delta))),
             'd_rho':float(np.sqrt(np.mean((delta/denominator)**2)))}
        trajectory.append(row);snapshots.append(working.copy())
    snapshot()
    while accepted<exchanges:
        attempts+=1
        if attempts>exchanges*20:raise RuntimeError('random target walk cannot generate enough valid moves')
        candidate=random_candidate(working,rng)
        if candidate is None:continue
        a,b,k,l=candidate;epsilon=float(rng.uniform(.05,.95)*min(working.mass[a],working.mass[b]))
        if epsilon<=1e-15:continue
        working.exchange(a,b,k,l,epsilon);working.validate(pi);accepted+=1
        if accepted%interval==0:snapshot()
    if trajectory[-1]['exchanges']!=accepted:snapshot()
    maximum=max(row['d_rho'] for row in trajectory)
    substantial=max(.5,.1*trajectory[0]['entropy'])
    eligible=[i for i,row in enumerate(trajectory) if row['d_rho']>=.8*maximum and row['entropy']>=trajectory[0]['entropy']+substantial]
    if not eligible:
        if exchanges<10000:
            # Same seed and trajectory prefix, explicitly report extension.
            return make_target(initial,pi,seed,exchanges=10000,interval=interval)
        raise RuntimeError('no substantially more entropic and separated target after 10000 moves')
    chosen=max(eligible,key=lambda i:(trajectory[i]['entropy'],-trajectory[i]['exchanges']))
    target=snapshots[chosen];sip=independent_sip(target);delta=abs((sip-original)[PAIRS]);rho=delta/denominator
    metadata={'H0':initial.entropy(),'Hstar':target.entropy(),'support0':len(initial.mass),'supportstar':len(target.mass),
        'selected_exchanges':trajectory[chosen]['exchanges'],'walk_exchanges':exchanges,'walk_attempts':attempts,
        'initial_RMSE':trajectory[chosen]['RMSE_from_initial'],'d_rho':trajectory[chosen]['d_rho'],
        'maximum_observed_d_rho':maximum,'entropy_increase':target.entropy()-initial.entropy(),
        'abs_SIP_difference':dict(zip(['min','median','max'],map(float,[np.min(delta),np.median(delta),np.max(delta)]))),
        'abs_rho_difference':dict(zip(['min','median','max'],map(float,[np.min(rho),np.median(rho),np.max(rho)]))),
        'validation':validate_design(target,pi,sip),'initial_validation':validate_design(initial,pi,original),
        'target_selection':'maximum entropy among snapshots with dependence distance >=80% of observed maximum and entropy increase >=max(0.5,0.1*H0)',
        'target_generator':'objective-blind uniformly chosen valid support-pair/unit exchanges; epsilon uniform in [0.05,0.95] times available mass'}
    return target,trajectory,metadata
