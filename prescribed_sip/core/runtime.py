"""Shared serial budgets, log-spaced checkpoints and run outcomes."""
from dataclasses import dataclass,field
from time import perf_counter
import numpy as np


class BudgetStop(RuntimeError): pass


@dataclass
class Outcome:
    support: object
    initial: dict
    final: dict
    runtime_seconds: float
    candidate_moves: int
    accepted_moves: int
    validation: dict
    checkpoints: list
    status: str = 'completed'
    failure_reason: str = ''
    details: dict = field(default_factory=dict)


class Monitor:
    def __init__(self,objective,initial,pi,evaluations,seconds):
        self.objective=objective; self.pi=pi
        self.max_evaluations=int(evaluations); self.max_seconds=float(seconds)
        self.started=perf_counter(); self.candidates=0; self.accepted=0
        self.checkpoints=[]; self.next_checkpoint=1
        self.best=initial.copy(); self.initial=objective.initialize(initial)
        self.best_metrics=self.initial.copy()
        self.validation=initial.validate(pi)
        self.validation['max_SIP_incremental_error']=0.
        self.record(initial,self.initial,force=True)

    def guard(self):
        if self.candidates>=self.max_evaluations: raise BudgetStop('evaluation_budget')
        if perf_counter()-self.started>=self.max_seconds: raise BudgetStop('time_budget')

    def validate(self,support,incremental_error=0.):
        vals=support.validate(self.pi)
        for k,v in vals.items():
            if k=='min_sample_probability': self.validation[k]=min(self.validation[k],v)
            else: self.validation[k]=max(self.validation[k],v)
        self.validation['max_SIP_incremental_error']=max(self.validation['max_SIP_incremental_error'],incremental_error)
        if incremental_error>1e-10: raise RuntimeError('incremental SIP drift')

    def record(self,support,metrics,force=False):
        if force or self.candidates>=self.next_checkpoint:
            extra = {}
            if getattr(self.objective, 'report_design_diagnostics', False):
                extra = {'entropy': support.entropy(),
                         'incumbent_support_size': len(self.best.mass),
                         'incumbent_entropy': self.best.entropy()}
            self.checkpoints.append({'candidate_moves':self.candidates,'accepted_moves':self.accepted,
                                     'runtime_seconds':perf_counter()-self.started,
                                     'support_size':len(support.mass),**metrics,**extra,
                                     **{'incumbent_'+k:v for k,v in self.best_metrics.items()}})
            self.next_checkpoint=max(self.next_checkpoint+1,int(np.ceil(max(1,self.candidates)*1.6)))

    def incumbent(self,support,metrics):
        if metrics['D2']<self.best_metrics['D2']:
            self.best=support.copy(); self.best_metrics=metrics.copy()

    def finish(self,support,status='completed',reason='',details=None):
        self.validate(support)
        self.objective.initialize(support)
        final=self.objective.metrics()
        self.record(support,final,force=True)
        return Outcome(support,self.initial,final,perf_counter()-self.started,self.candidates,
                       self.accepted,self.validation,self.checkpoints,status,reason,details or {})
