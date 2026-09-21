"""Independent fixed-size target generators; optimizers receive moments only."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import comb
import json
import numpy as np
from scipy.special import logsumexp
from prescribed_sip.core.design import Support
from prescribed_sip.core.objective import PairObjective
from prescribed_sip.methods.exact import space_for
from prescribed_sip.methods.dsd import projection_factor,projection_moments


def seed_for(master,*keys):
    encoded=json.dumps([int(master),*keys],separators=(',',':'),ensure_ascii=True).encode()
    return int.from_bytes(sha256(encoded).digest()[:8],'little')%(2**63-1)


def bounded_pps(size,n):
    size=np.asarray(size,dtype=np.float64)
    if np.any(size<=0) or n<1 or n>len(size): raise ValueError('positive sizes and valid n required')
    pi=np.zeros_like(size); live=np.ones(len(size),bool); remaining=float(n)
    while np.any(live):
        candidate=remaining*size[live]/size[live].sum()
        certain=candidate>=1
        ids=np.flatnonzero(live)
        if not np.any(certain): pi[ids]=candidate; break
        pi[ids[certain]]=1.; live[ids[certain]]=False; remaining-=sum(certain)
    return pi


@dataclass
class Target:
    N:int
    n:int
    pi:np.ndarray
    sip:object
    coords:np.ndarray
    kind:str
    fip_type:str
    feasible_known:object
    metadata:dict


def selected_pairs(N,density,structure,coords,rng):
    M=N*(N-1)//2; count=max(1,round(M*density))
    if density>=1: return list(combinations(range(N),2))
    if structure=='random':
        # O(number of chosen pairs); no N by N allocation for sparse targeting.
        pairs=set()
        while len(pairs)<count:
            i,j=sorted(rng.choice(N,2,replace=False).tolist()); pairs.add((i,j))
        return sorted(pairs)
    if structure=='nearby':
        # Line: exact closest pairs. Planar: exact distances with only one row
        # workspace and an O(M) list (no dense matrix), then retain chosen pairs.
        import heapq
        def distances():
            for i in range(N-1):
                dist=np.sum((coords[i+1:]-coords[i])**2,axis=1)
                for t,d in enumerate(dist): yield (float(d),i,i+t+1)
        return sorted((i,j) for d,i,j in heapq.nsmallest(count,distances()))
    raise ValueError('unknown pair structure')


def cp_fip(odds,n):
    odds=np.asarray(odds,dtype=np.float64); N=len(odds)
    if n==0: return np.zeros(N)
    if n==N: return np.ones(N)
    logodds=np.log(odds)
    pre=np.full((N+1,n+1),-np.inf); post=pre.copy()
    pre[0,0]=0.; post[N,0]=0.
    for k in range(N):
        pre[k+1]=pre[k]
        pre[k+1,1:]=np.logaddexp(pre[k,1:],logodds[k]+pre[k,:-1])
    for k in range(N-1,-1,-1):
        post[k]=post[k+1]
        post[k,1:]=np.logaddexp(post[k+1,1:],logodds[k]+post[k+1,:-1])
    logZ=pre[N,n]
    return np.array([np.exp(logodds[k]+logsumexp(pre[k,:n]+post[k+1,n-1::-1])-logZ) for k in range(N)])


def cp_moments(odds,n,pairs):
    # P(j included | i included) is CP(n-1) on U\{i}; exact log-DP.
    pi=cp_fip(odds,n); grouped={}
    for i,j in pairs: grouped.setdefault(i,[]).append(j)
    sip={}
    for i,js in grouped.items():
        others=np.delete(odds,i); conditional=cp_fip(others,n-1)
        for j in js: sip[(i,j)]=float(pi[i]*conditional[j-1])
    return pi,sip


def make_target(N,n,kind,seed,geometry='planar',density=1.,pair_structure='random',**kwargs):
    rng=np.random.default_rng(seed)
    coords_rng=np.random.default_rng(seed_for(seed,'coordinates'))
    pair_rng=np.random.default_rng(seed_for(seed,'pairs'))
    coords=np.column_stack([np.linspace(0,1,N),np.zeros(N)]) if geometry=='1d' else coords_rng.uniform(0,1,(N,2))
    diagnostic_support=kwargs.get('diagnostic_support',False)
    pairs=[] if diagnostic_support else selected_pairs(N,density,pair_structure,coords,pair_rng)
    metadata={'target_seed':seed,'geometry':geometry,'pair_structure':pair_structure,'density':density,
              'generator':'independent of GFS search','hidden_design_discarded':True}
    feasible=True; fip_type='equal'; pi=np.full(N,n/N)
    if kind=='srs':
        vals={p:n*(n-1)/(N*(N-1)) for p in pairs}
    elif kind=='stratified':
        # Balanced integer sample allocations to contiguous strata along ordering.
        H=min(n,4); groups=np.array_split(np.arange(N),H)
        allocations=np.full(H,n//H); allocations[:n%H]+=1
        labels=np.empty(N,int)
        for h,g in enumerate(groups): labels[g]=h; pi[g]=allocations[h]/len(g)
        vals={}
        for i,j in pairs:
            h=labels[i]
            vals[(i,j)]=float(allocations[h]*(allocations[h]-1)/(len(groups[h])*(len(groups[h])-1))) if h==labels[j] else float(pi[i]*pi[j])
        fip_type='equal' if np.ptp(pi)<1e-12 else 'unequal'
    elif kind in ('conditional_poisson','cp_unequal'):
        odds=rng.lognormal(0,.8,N)
        pi,vals=cp_moments(odds,n,pairs); fip_type='unequal'
        metadata.update({'odds':odds.tolist(),'FIP_generator':'design-induced exact conditional-Poisson moments'})
    elif kind=='dsd':
        sizes=rng.gamma(3,1,N); pi=bounded_pps(sizes,n); fip_type='unequal'
        V=projection_factor(pi); vals=projection_moments(V,pairs)
        metadata['FIP_generator']='bounded PPS from gamma size measures'
    elif kind in ('infeasible_small','perturbed_small','perturbed_moderate','perturbed_large'):
        value=n*(n-1)/(N*(N-1)); vals={p:value for p in pairs}
        magnitude={'infeasible_small':value+.01,'perturbed_small':.01,'perturbed_moderate':.05,'perturbed_large':.2}[kind]
        for p,sgn in [((0,1),1),((2,3),1),((0,2),-1),((1,3),-1)]:
            if p in vals: vals[p]+=sgn*magnitude
        feasible=False if kind=='infeasible_small' else None
        metadata.update({'perturbation':magnitude,'row_balanced_four_cycle':True})
    else:
        if N<=14 and kind in ('arbitrary','sparse_support','diffuse','unequal'):
            space=space_for(N,n); samples=space.samples
            if kind=='sparse_support':
                ids=rng.choice(len(samples),min(len(samples),max(N,2*n)),replace=False)
                samples=[samples[t] for t in ids]
            missing=set(range(N))-set(i for s in samples for i in s)
            for i in sorted(missing):
                rest=rng.choice(np.delete(np.arange(N),i),n-1,replace=False)
                samples=[*samples,tuple(sorted([i,*rest.tolist()]))]
            metadata['coverage_augmented_samples']=len(missing)
            alpha=10. if kind=='diffuse' else .5
            p=rng.dirichlet(np.full(len(samples),alpha))
            mass=dict(zip(samples,p))
            d=Support(N,n,mass); pi=d.fip(); fip_type='unequal'
        elif kind=='random_equal':
            # Mixture of cyclic orbits of uniformly chosen fixed-size samples.
            # Each complete orbit has exactly equal FIP; not a graphical optimizer.
            mass={}
            for _ in range(3):
                base=rng.choice(N,n,replace=False)
                for shift in range(N):
                    s=tuple(sorted(((base+shift)%N).tolist()))
                    mass[s]=mass.get(s,0.)+1/(3*N)
            d=Support(N,n,mass); pi=d.fip()
        elif kind in ('random_unequal','sparse_large'):
            count=max(2*N//n,30) if kind=='sparse_large' else max(10*N//n,100)
            samples=[tuple(sorted(rng.choice(N,n,replace=False).tolist())) for _ in range(count)]
            covered=set(i for s in samples for i in s)
            for i in sorted(set(range(N))-covered):
                rest=rng.choice(np.delete(np.arange(N),i),n-1,replace=False)
                samples.append(tuple(sorted([i,*rest.tolist()])))
            p=rng.dirichlet(np.full(len(samples),.5 if kind=='sparse_large' else 2.))
            mass={}
            for s,v in zip(samples,p): mass[s]=mass.get(s,0.)+v
            d=Support(N,n,mass); pi=d.fip(); fip_type='unequal'
        else: raise ValueError(f'unknown target kind: {kind}')
        d.validate(pi)
        if diagnostic_support: return d
        temp=PairObjective(N,{p:0. for p in pairs})
        moments=d.pair_values(temp); vals=dict(zip(pairs,moments))
        metadata.update({'hidden_support_size':len(d.mass),'hidden_entropy':d.entropy(),
                         'FIP_generator':'independent design-induced moments'})
    if density>=1:
        sip=np.zeros((N,N),dtype=np.float64)
        for (i,j),v in vals.items(): sip[i,j]=sip[j,i]=v
        np.fill_diagonal(sip,pi)
    else: sip=vals
    return Target(N,n,pi,sip,coords,kind,fip_type,feasible,metadata)
