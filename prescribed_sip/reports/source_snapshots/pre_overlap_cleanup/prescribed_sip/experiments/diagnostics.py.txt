"""Independent full-pair diagnostics outside search budgets, with row workspaces."""
from time import perf_counter
import numpy as np
from scipy import sparse
from prescribed_sip.experiments.targets import make_target,cp_fip
from prescribed_sip.methods.dsd import projection_factor


def support_rows(design):
    rows=[];columns=[]
    for k,s in enumerate(design.mass):
        rows.extend([k]*len(s));columns.extend(s)
    x=sparse.csr_matrix((np.ones(len(rows)),(rows,columns)),shape=(len(design.mass),design.N))
    weighted=x.multiply(np.array(list(design.mass.values()))[:,None]).tocsr()
    return lambda ids: (x[:,ids].T@weighted).toarray()


def sparse_full_diagnostics(target,initial,final,method,block_size=32):
    started=perf_counter();N=target.N;n=target.n
    initial_rows=support_rows(initial)
    if final is not None: final_rows=support_rows(final)
    elif method=='DSD-structured':
        v=projection_factor(target.pi)
        final_rows=lambda ids: target.pi[ids,None]*target.pi[None,:]-(v[ids]@v.T)**2
    else: raise ValueError('full diagnostics require a support or analytic DSD')
    kind=target.kind
    if kind=='srs': target_rows=lambda ids: np.full((len(ids),N),n*(n-1)/(N*(N-1)))
    elif kind=='dsd':
        vtarget=projection_factor(target.pi)
        target_rows=lambda ids: target.pi[ids,None]*target.pi[None,:]-(vtarget[ids]@vtarget.T)**2
    elif kind=='stratified':
        H=min(n,4);groups=np.array_split(np.arange(N),H);labels=np.empty(N,int)
        allocation=np.full(H,n//H);allocation[:n%H]+=1
        for h,g in enumerate(groups): labels[g]=h
        def target_rows(ids):
            result=target.pi[ids,None]*target.pi[None,:]
            for k,i in enumerate(ids):
                h=labels[i];L=len(groups[h])
                result[k,groups[h]]=allocation[h]*(allocation[h]-1)/(L*(L-1)) if L>1 else 0.
            return result
    elif kind in ('conditional_poisson','cp_unequal'):
        odds=np.array(target.metadata['odds'])
        def target_rows(ids):
            result=np.zeros((len(ids),N))
            for k,i in enumerate(ids): result[k,np.arange(N)!=i]=target.pi[i]*cp_fip(np.delete(odds,i),n-1)
            return result
    else:
        hidden=make_target(N,n,kind,target.metadata['target_seed'],target.metadata['geometry'],
                           density=target.metadata['density'],pair_structure=target.metadata['pair_structure'],diagnostic_support=True)
        if np.max(abs(hidden.fip()-target.pi))>1e-10: raise RuntimeError('diagnostic target regeneration mismatch')
        target_rows=support_rows(hidden)
    sums=np.zeros(2);maxima=np.zeros(2);workspace=0
    for start in range(0,N,block_size):
        ids=np.arange(start,min(N,start+block_size));truth=target_rows(ids)
        mask=np.arange(N)[None,:]>ids[:,None]
        for k,rows in enumerate((initial_rows,final_rows)):
            current=rows(ids);residual=(current-truth)[mask]
            sums[k]+=residual@residual;maxima[k]=max(maxima[k],float(np.max(abs(residual),initial=0)))
            workspace=max(workspace,truth.nbytes+current.nbytes+mask.nbytes+residual.nbytes)
    M=N*(N-1)/2
    return {'initial_RMSE2':float(np.sqrt(sums[0]/M)),'final_RMSE2':float(np.sqrt(sums[1]/M)),
            'initial_Dinf':float(maxima[0]),'final_Dinf':float(maxima[1]),
            'full_diagnostics_seconds':perf_counter()-started,'full_diagnostics_max_workspace_bytes':workspace}
