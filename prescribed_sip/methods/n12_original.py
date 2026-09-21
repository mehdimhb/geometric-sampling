"""Original GBFS, with profiling and same-bytecode precision sensitivity."""
from contextlib import contextmanager
import inspect
import resource
from time import perf_counter
from types import FunctionType
import numpy as np
from graphical_sampling.search import GreedyBestFirstSearch
from prescribed_sip.methods.package_gfs import initial_design,extract_support,seeded_package,Float64Design
from prescribed_sip.methods.exact import space_for
from prescribed_sip.core.design import InvariantError
from prescribed_sip.experiments.n12_targets import independent_sip,validate_design,PAIRS


class CandidateBudget(RuntimeError):pass


def precision_run(original):
    namespace=dict(original.__globals__)
    namespace['round']=lambda value,ndigits=None:float(value)
    clone=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    clone.__kwdefaults__=original.__kwdefaults__
    return clone


def run_original(pi,target,seed,evaluations=5000,high_precision=False,record_interval=50):
    started=perf_counter();space=space_for(12,4);lookup={s:i for i,s in enumerate(space.samples)}
    units=space.units.toarray();pairs=space.pairs.toarray();target_values=target[PAIRS]
    profile={'criterion_seconds':0.,'SIP_computation_seconds':0.,'support_extraction_seconds':0.,
             'candidate_generation_seconds':0.,'design_copy_seconds':0.,'design_update_seconds':0.,'validation_seconds':0.}
    count=0;best_support=None;best_loss=None;last_improvement=0;trace=[];initial_loss=None
    worst={'max_FIP_error':0.,'max_probability_sum_error':0.,'min_sample_probability':1.,'max_SIP_recompute_error':0.}
    comparator=(lambda v:v) if high_precision else (lambda v:round(v,9))
    def vector(support):
        probabilities=np.zeros(495,dtype=np.float64)
        for sample,p in support.mass.items():
            if sample not in lookup or p < -1e-12:raise InvariantError('invalid scored sampling design')
            probabilities[lookup[sample]]=p
        return probabilities
    def metrics(support,loss,values):
        return {'D2':float(loss),'RMSE':float(np.sqrt(loss/66)),
                'Dinf':float(np.max(abs(values-target_values))),
                'recovery_percent':float(100*(1-np.sqrt(loss/initial_loss))) if initial_loss>0 else 100.}
    def checkpoint(force=False):
        if force or count%record_interval==0:
            probabilities=vector(best_support);values=pairs@probabilities
            row={'candidates':count,'elapsed_seconds':perf_counter()-started,
                 **metrics(best_support,best_loss,values)}
            if trace and trace[-1]['candidates']==count:trace[-1]=row
            else:trace.append(row)
    def criterion(design):
        nonlocal count,best_support,best_loss,initial_loss,last_improvement
        tick=perf_counter();is_initial=design is initial
        if not is_initial:
            if count>=evaluations:raise CandidateBudget('scored_candidate_budget')
            count+=1
        stage=perf_counter();support=extract_support(design);profile['support_extraction_seconds']+=perf_counter()-stage
        probabilities=vector(support)
        stage=perf_counter();fip=units@probabilities;mass_error=abs(float(probabilities.sum())-1);fip_error=float(np.max(abs(fip-pi)))
        if max(mass_error,fip_error)>1e-10:raise InvariantError('scored design violates FIP/mass')
        worst['max_FIP_error']=max(worst['max_FIP_error'],fip_error)
        worst['max_probability_sum_error']=max(worst['max_probability_sum_error'],mass_error)
        worst['min_sample_probability']=min(worst['min_sample_probability'],min(support.mass.values()))
        profile['validation_seconds']+=perf_counter()-stage
        stage=perf_counter();values=pairs@probabilities;difference=values-target_values;loss=float(difference@difference)
        profile['SIP_computation_seconds']+=perf_counter()-stage
        if initial_loss is None:initial_loss=loss
        if best_loss is None or comparator(loss)<comparator(best_loss):
            best_loss=loss;best_support=support.copy();last_improvement=count
        checkpoint()
        profile['criterion_seconds']+=perf_counter()-tick
        return loss
    class InstrumentedSearch(GreedyBestFirstSearch):
        def _exchange_neighbors(self,*args,**kwargs):
            iterator=super()._exchange_neighbors(*args,**kwargs)
            while True:
                tick=perf_counter()
                try:value=next(iterator)
                except StopIteration:return
                finally:profile['candidate_generation_seconds']+=perf_counter()-tick
                yield value
        def _order_neighbors(self,*args,**kwargs):
            iterator=super()._order_neighbors(*args,**kwargs)
            while True:
                tick=perf_counter()
                try:value=next(iterator)
                except StopIteration:return
                finally:profile['candidate_generation_seconds']+=perf_counter()-tick
                yield value
    if high_precision:InstrumentedSearch.run=precision_run(GreedyBestFirstSearch.run)
    original_copy=Float64Design.copy;original_exchange=Float64Design.exchange
    def timed_copy(self,*args,**kwargs):
        tick=perf_counter()
        try:return original_copy(self,*args,**kwargs)
        finally:profile['design_copy_seconds']+=perf_counter()-tick
    def timed_exchange(self,*args,**kwargs):
        tick=perf_counter()
        try:return original_exchange(self,*args,**kwargs)
        finally:profile['design_update_seconds']+=perf_counter()-tick
    Float64Design.copy=timed_copy;Float64Design.exchange=timed_exchange
    status='search_completed';reason='frontier exhausted or iteration cap';search=None
    options={'max_iterations':max(1000,evaluations*2),'max_open_set_size':50,'top_k':1,
             'num_new_order_nodes':2,'num_new_exchange_nodes':10,'num_clusters_range':(1,1),
             'num_zones_range':(1,1),'num_changes_range':(1,1),'num_zone_changes':0,
             'pull_strategy':'default','exchange_coef':.75,'num_explore':1,'n_jobs':1,
             'backtrack_depth':0,'stagnation_limit':100,'output_mode':'silent'}
    try:
        with seeded_package(seed):
            coords=np.column_stack([np.linspace(0,1,12),np.zeros(12)])
            initial=initial_design(pi,coords)
            search=InstrumentedSearch([initial],criterion)
            def progress(obj,iteration):
                if count-last_improvement>=2000:raise CandidateBudget('no incumbent improvement for 2000 scored candidates')
            search.run(**options,stop_when=lambda d,value:value<66*1e-20,progress_callback=progress)
            if best_loss<66*1e-20:status='recovered';reason='RMSE below 1e-10'
    except CandidateBudget as error:status='budget_exhausted' if count>=evaluations else 'stagnated';reason=str(error)
    finally:Float64Design.copy=original_copy;Float64Design.exchange=original_exchange
    final=best_support
    final_sip=independent_sip(final);loss=float(np.sum((final_sip[PAIRS]-target_values)**2))
    validation=validate_design(final,pi,final_sip)
    validation['max_SIP_recompute_error']=float(np.max(abs((pairs@vector(final))-final_sip[PAIRS])))
    worst['max_SIP_recompute_error']=validation['max_SIP_recompute_error']
    validation.update({'maximum_scored_design_'+k:v for k,v in worst.items()})
    best_loss=loss;checkpoint(force=True)
    runtime=perf_counter()-started
    row={'method':'Original GFS high precision' if high_precision else 'Original GFS','seed':int(seed),
         'status':status,'failure_reason':reason,'final_D2':loss,'final_RMSE':float(np.sqrt(loss/66)),
         'final_Dinf':float(np.max(abs(final_sip[PAIRS]-target_values))),
         'recovery_percent':float(100*(1-np.sqrt(loss/initial_loss))),
         'runtime_seconds':runtime,'candidate_evaluations':count,'validation':validation,'trace':trace,
         'details':{'score':'D2','score_rounding_decimals':None if high_precision else 9,
             'same_original_run_bytecode':InstrumentedSearch.run.__code__ is GreedyBestFirstSearch.run.__code__,
             'search_options':options,'last_improvement_candidate':last_improvement,
             'no_improvement_stop_candidates':2000,'iterations':getattr(search,'iterations_completed',0),
             'profile':profile,'evaluations_per_second':count/runtime,
             'average_candidate_criterion_seconds':profile['criterion_seconds']/max(1,count),
             'peak_process_RSS_MB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             'memory_measurement':'process-lifetime RSS high water; not incremental method memory',
             'high_precision_implementation':'same run bytecode; isolated round binding returns raw float' if high_precision else 'unmodified run and builtin round',
             'initial_D2':initial_loss,'initial_RMSE':float(np.sqrt(initial_loss/66))}}
    return row,final
