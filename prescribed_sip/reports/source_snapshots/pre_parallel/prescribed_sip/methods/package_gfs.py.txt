"""Precision/RNG/budget adapter for the unmodified package GBFS algorithm.

No search policy is replaced. The package's nine-decimal comparisons, reservoir,
neighbor generation, duplicate pruning and frontier management are retained.
The adapter keeps masses down to 1e-15 instead of the package's 1e-9 push cutoff.
"""
from contextlib import contextmanager
from math import fsum
import random
import numpy as np
from graphical_sampling import Design,Population,Order
from graphical_sampling.order import Cluster,Zone
from graphical_sampling.search import GreedyBestFirstSearch
from prescribed_sip.core.design import Support,InvariantError
from prescribed_sip.core.runtime import Monitor,BudgetStop


@contextmanager
def seeded_package(seed):
    """Serial-only transaction for package functions that create unseeded RNGs."""
    original=np.random.default_rng
    random_state=random.getstate(); numpy_state=np.random.get_state()
    root=original(seed)
    def factory(s=None):
        return original(int(root.integers(0,2**63-1)) if s is None else s)
    np.random.default_rng=factory
    random.seed(int(seed)); np.random.seed(int(seed)%(2**32))
    try: yield
    finally:
        np.random.default_rng=original
        random.setstate(random_state); np.random.set_state(numpy_state)


class Float64Design(Design):
    def _build(self):
        # A certainty interval wraps onto itself. Original event remainders can
        # differ by roundoff, so start/end sorting may incorrectly remove it.
        # Condition out exact certainty units; reuse the original builder intact
        # on the remaining population, then append them to every support sample.
        certain=np.flatnonzero(self.pop.inclusions==1.)
        if not len(certain):
            return super()._build()
        from graphical_sampling.structs import _Sample,_MaxHeap
        original_pop,original_order=self._pop,self._order
        live=np.flatnonzero(original_pop.inclusions<1.)
        remaining=original_pop.n-len(certain)
        if remaining==0:
            self.events=[]
            for part in range(self.num_partitions):
                self._push(part,_Sample(1./self.num_partitions,frozenset(certain.tolist())))
            return
        reduced_pop=Population(original_pop.coords[live],original_pop.inclusions[live],
                               variable=np.ones(len(live)),n=None)
        reduced_pop._n=remaining
        mapping={int(unit):i for i,unit in enumerate(live)}
        data=original_order.get()
        kept=[(mapping[int(unit)],share) for unit,share in data if int(unit) in mapping]
        reduced_order=Order(reduced_pop)
        reduced_order._order=np.asarray(kept,dtype=np.float64)
        reduced_order.num_zones=original_order.num_zones
        reduced_order.num_splits=original_order.num_splits
        try:
            self._pop,self._order=reduced_pop,reduced_order
            super()._build()
            heaps=[]
            fixed=set(certain.tolist())
            for heap in self._heaps:
                rebuilt=_MaxHeap()
                for sample in heap:
                    ids=frozenset(fixed|{int(live[i]) for i in sample.ids})
                    rebuilt.push(_Sample(float(sample.prob),ids))
                heaps.append(rebuilt)
            self._heaps=heaps
        finally:
            self._pop,self._order=original_pop,original_order

    @property
    def all_samples_and_probs(self):
        if self._all_samples_and_probs is not None:
            return self._all_samples_and_probs
        support=extract_support(self)
        for s in support.mass:
            if len(s)!=self.pop.n: raise InvariantError('package sample has wrong size')
        samples=np.array(list(support.mass),dtype=np.int64)
        probs=np.array(list(support.mass.values()),dtype=np.float64)
        self._all_samples_and_probs=(samples,probs)
        return self._all_samples_and_probs

    def copy(self):
        clone=super().copy()
        clone.__class__=Float64Design
        for heap in clone._heaps:
            heap.rng=np.random.default_rng()
        # Base Design.copy creates a new unseeded generator; transaction seeds it.
        return clone

    def _push(self,part_idx,*args):
        # Numerical-layer adaptation only: do not drop scientifically material mass.
        for sample in args:
            if sample.prob < -1e-15:
                raise InvariantError('negative package fragment')
            if sample.prob>1e-15:
                self._heaps[part_idx].push(sample)


def extract_support(design):
    pieces={}
    for heap in design._heaps:
        for sample in heap:
            ids=tuple(sorted(sample.ids))
            pieces.setdefault(ids,[]).append(float(sample.prob))
    return Support(design.pop.N,design.pop.n,{s:fsum(p) for s,p in pieces.items()})


def initial_design(pi,coords,permutation=None):
    pi=np.asarray(pi,dtype=np.float64)
    n=int(round(pi.sum())); N=len(pi)
    pop=Population(np.asarray(coords,dtype=np.float64),pi,variable=np.ones(N),n=None)
    # Population's unrounded sum-to-int inference can truncate 2.9999999999999996.
    pop._n=n
    indices=np.arange(N) if permutation is None else np.asarray(permutation,dtype=int)
    zone=Zone(_shares=np.ones(N),_indices=indices,sort=list(range(N)))
    order=Order(pop); order.clusters=[Cluster(label=0,zones=[zone])]
    order.num_zones=1; order.num_splits=1; order._build_order(1)
    d=Float64Design.from_order(pop,order)
    extract_support(d).validate(pi)
    return d


def run_package(design,objective,pi,seed,evaluations,seconds,batch=10,**kwargs):
    support=extract_support(design)
    monitor=Monitor(objective,support,pi,evaluations,seconds)
    initial_supports={id(design)}
    best_design=design; best_value=monitor.initial['target_RMSE2']
    attempts=0
    # Criterion instrumentation observes all fully evaluated candidates. It uses the
    # exact same nine-decimal incumbent comparison as the package reservoir.
    def criterion(d):
        nonlocal best_design,best_value
        is_initial=id(d) in initial_supports
        if not is_initial:
            monitor.guard(); monitor.candidates+=1
        s=extract_support(d); monitor.validate(s)
        objective.initialize(s); metrics=objective.metrics(); val=metrics['target_RMSE2']
        if round(val,9)<round(best_value,9):
            best_design=d; best_value=val; monitor.accepted+=1
            monitor.best=s.copy(); monitor.best_metrics=metrics.copy()
        monitor.record(s,metrics)
        return val
    search=None; status='completed'; reason=''
    with seeded_package(seed):
        design=design.copy()
        initial_supports={id(design)}
        best_design=design
        try:
            search=GreedyBestFirstSearch([design],criterion)
            def progress(obj,iteration): monitor.guard()
            # Single-zone general GFS: original order changes and mass exchanges.
            search.run(max_iterations=max(10,evaluations*2),max_open_set_size=50,top_k=1,
                       num_new_order_nodes=max(1,batch//5),num_new_exchange_nodes=batch,
                       num_clusters_range=(1,1),num_zones_range=(1,1),num_changes_range=(1,1),
                       num_zone_changes=0,pull_strategy='default',exchange_coef=.75,
                       n_jobs=1,backtrack_depth=0,stagnation_limit=10,output_mode='silent',
                       progress_callback=progress,
                       stop_when=lambda d,v:v<1e-10)
        except BudgetStop as e:
            reason=str(e); status='timeout' if reason=='time_budget' else 'budget_exhausted'
        except (InvariantError,ValueError,RuntimeError) as e:
            status='numerical_failure' if isinstance(e,InvariantError) else 'implementation_failure'
            reason=f'{type(e).__name__}: {e}'
    final=extract_support(best_design)
    details={'search_implementation':'graphical_sampling.search.GreedyBestFirstSearch.run',
             'package_iterations':getattr(search,'iterations_completed',0),
             'accepted_definition':'nine-decimal best-evaluated incumbent improvements',
             'precision_adapter_push_cutoff':1e-15,'package_score_rounding_decimals':9,
             'search_options':{'batch':batch,'exchange_coef':.75,'frontier_size':50,
                               'order_candidates':max(1,batch//5),'backtrack_depth':0}}
    return monitor.finish(final,status,reason,details)
