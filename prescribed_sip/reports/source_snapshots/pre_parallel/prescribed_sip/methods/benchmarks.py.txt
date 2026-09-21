"""Structured competitor adapters with explicit applicability and no hidden fitting."""
from itertools import combinations
from time import perf_counter
from math import comb
import numpy as np
from prescribed_sip.methods.dsd import projection_factor,projection_moments
from prescribed_sip.methods.sinha import sinha_parameters,sinha_moments,sinha_support
from prescribed_sip.core.runtime import Outcome
from prescribed_sip.core.design import Support


def structured_outcome(target,objective,initial,method):
    start=perf_counter(); initial_metrics=objective.initialize(initial); pi=target.pi; N=target.N; n=target.n
    pairs=list(zip(objective.i.tolist(),objective.j.tolist()))
    support=None; details={}; status='completed'; reason=''
    if method=='DSD-structured':
        V=projection_factor(pi)
        values=projection_moments(V,pairs)
        max_fip=float(np.max(abs(np.sum(V*V,axis=1)-pi)))
        details={'structured_baseline':True,'kernel_order':'original population ordering',
                 'factor_memory_bytes':V.nbytes,'no_arbitrary_SIP_fit':True}
        if N<=14:
            K=V@V.T; mass={}
            for s in combinations(range(N),n):
                p=float(np.linalg.det(K[np.ix_(s,s)]))
                if p>1e-15: mass[s]=p
            support=Support(N,n,mass)
    elif method=='Sinha':
        if not objective.full:
            raise NotImplementedError('documented direct Sinha inversion requires a full SIP target')
        sip={(i,j):float(target.sip[i,j]) for i,j in pairs}
        b=sinha_parameters(N,n,pi,sip)
        if min(b.values()) < -1e-12:
            raise NotImplementedError('target has negative Sinha pair-removal coefficients')
        # Roundoff-level negatives alone are removed and the correction is reported.
        correction=max(0.,-min(b.values()))
        b={p:max(v,0.) for p,v in b.items()}
        fitted_pi,values=sinha_moments(N,n,b)
        max_fip=float(np.max(abs(fitted_pi-pi)))
        details={'negative_coefficient_roundoff_correction':correction,'structured_baseline':False,
                 'specification':'Bondesson, Grafstrom & Traat 2014 pp183-185, type(b) direct inversion'}
        if N<=14: support=sinha_support(N,n,b)
    else: raise ValueError('unknown structured benchmark')
    current=np.array([values[p] for p in pairs]); objective.current=current
    objective.residual=current-objective.target; objective.D2=float(objective.residual@objective.residual)
    final=objective.metrics()
    validation={'max_FIP_error':max_fip,'max_probability_sum_error':0.,
                'min_sample_probability':float('nan'),'max_SIP_incremental_error':0.}
    if support is not None:
        vals=support.validate(pi); validation.update(vals)
    if max_fip>1e-10: raise RuntimeError('structured benchmark FIP validation failed')
    elapsed=perf_counter()-start
    checkpoints=[{'candidate_moves':0,'accepted_moves':0,'runtime_seconds':elapsed,
                  'support_size':len(support.mass) if support else float('nan'),**final,
                  **{'incumbent_'+k:v for k,v in final.items()}}]
    return Outcome(support,initial_metrics,final,elapsed,0,0,validation,checkpoints,status,reason,details)
