"""Serial representation/objective profiles; run after small-N validation."""
import argparse
import cProfile
import json
import pstats
import sys
from pathlib import Path
from time import perf_counter
from io import StringIO
import numpy as np
from prescribed_sip.core.objective import PairObjective


def representation_benchmark(N,n,repeats=1000,seed=20260918):
    rng=np.random.default_rng(seed)
    tuples=[tuple(sorted(rng.choice(N,n,replace=False).tolist())) for _ in range(repeats+1)]
    masks=[sum(1<<i for i in s) for s in tuples]
    sets=[frozenset(s) for s in tuples]
    rows=[]
    for name,values in [('tuple',tuples),('integer_mask',masks),('frozenset',sets)]:
        start=perf_counter(); checksum=0
        for a,b in zip(values[:-1],values[1:]):
            checksum+=(a^b).bit_count() if name=='integer_mask' else len(set(a)^set(b)) if name=='tuple' else len(a^b)
        elapsed=perf_counter()-start
        memory=sum(sys.getsizeof(v) for v in values)
        if name!='integer_mask': memory+=sum(sys.getsizeof(i) for v in values for i in v)
        rows.append({'N':N,'n':n,'representation':name,'repeats':repeats,
                     'elapsed_seconds':elapsed,'memory_bytes':memory,'difference_checksum':checksum})
    return rows


def profile(output):
    from prescribed_sip.methods.package_gfs import seeded_package,initial_design,extract_support,run_package
    from prescribed_sip.methods.exchange_search import run_exchange
    results=[]
    for N,n in [(250,5),(500,25),(1000,50),(2000,100)]:
        results.extend(representation_benchmark(N,n))
    objective_rows=[]
    for N in [250,500,1000,2000]:
        M=N*(N-1)//2
        for density in [1.,.25,.1,.01]:
            count=max(1,round(M*density))
            if density==1.: obj=PairObjective(N,np.zeros((N,N)))
            else:
                # Same population, selected pair storage only; avoid full-matrix allocation.
                pairs={}
                for i in range(N):
                    for j in range(i+1,N):
                        pairs[(i,j)]=0.
                        if len(pairs)==count: break
                    if len(pairs)==count: break
                obj=PairObjective(N,pairs)
            allocated=sum(v.nbytes for v in [obj.i,obj.j,obj.target,obj.current,obj.residual])
            if obj.lookup is not None:
                allocated+=sys.getsizeof(obj.lookup)+sum(sys.getsizeof(k)+sys.getsizeof(v) for k,v in obj.lookup.items())
            objective_rows.append({'N':N,'density':density,'pairs':count,'approx_objective_memory_bytes':allocated})
    pi=np.full(250,5/250); coords=np.column_stack([np.arange(250),np.zeros(250)])
    with seeded_package(1):d=initial_design(pi,coords)
    target=np.full((250,250),5*4/(250*249)); np.fill_diagonal(target,pi)
    timings={}
    for method in ['Package-GFS','GFS-Directed']:
        profiler=cProfile.Profile();profiler.enable()
        if method=='Package-GFS': outcome=run_package(d,PairObjective(250,target),pi,seed=4,evaluations=100,seconds=10)
        else: outcome=run_exchange(extract_support(d),PairObjective(250,target),pi,seed=4,evaluations=100,seconds=10,method=method)
        profiler.disable(); stream=StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(20)
        timings[method]={'runtime_seconds':outcome.runtime_seconds,'candidate_moves':outcome.candidate_moves,
                         'status':outcome.status,'profile':stream.getvalue()}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'representation_benchmark':results,'objective_storage':objective_rows,'search_profiles':timings},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='prescribed_sip/reports/profile.json')
    profile(parser.parse_args().output)
