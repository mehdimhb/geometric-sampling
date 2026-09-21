"""Double precision active-support designs and exact two-sample exchanges."""
from collections import defaultdict
from itertools import combinations
from math import fsum
import numpy as np


class InvariantError(RuntimeError):
    """A design failed a scientifically required constraint."""


def exchange_direction(a, b, k, l):
    a, b = frozenset(a), frozenset(b)
    if k not in a-b or l not in b-a:
        raise ValueError('k and l must belong to opposite sample differences')
    c = defaultdict(int)
    for r in (a ^ b) - {k, l}:
        sign = int(r in b) - int(r in a)
        c[tuple(sorted((k,r)))] += sign
        c[tuple(sorted((l,r)))] -= sign
    return {pair:value for pair,value in c.items() if value}


class Support:
    def __init__(self, N, n, mass):
        self.N, self.n = int(N), int(n)
        self.mass = {}
        for s,p in mass.items():
            sample=tuple(sorted(s))
            self.mass[sample]=self.mass.get(sample,0.)+float(p)
        self.pruned_mass = 0.

    def copy(self):
        clone=Support(self.N,self.n,self.mass)
        clone.pruned_mass=self.pruned_mass
        return clone

    def fip(self):
        pi=np.zeros(self.N,dtype=np.float64)
        for s,p in self.mass.items():
            pi[np.asarray(s,dtype=int)] += p
        return pi

    def pair_values(self, objective):
        values=np.zeros(len(objective.target),dtype=np.float64)
        for s,p in self.mass.items():
            indices=objective.sample_indices(s)
            values[indices] += p
        return values

    def validate(self, target_pi, tolerance=1e-10):
        if not self.mass:
            raise InvariantError('empty support')
        for s,p in self.mass.items():
            if len(s)!=self.n or len(set(s))!=self.n or any(i<0 or i>=self.N for i in s):
                raise InvariantError('invalid sample size or unit identifier')
            if not np.isfinite(p) or p < -tolerance:
                raise InvariantError('negative or nonfinite sample probability')
        pi=self.fip()
        summary={'max_probability_sum_error':abs(fsum(self.mass.values())-1.),
                 'max_FIP_error':float(np.max(np.abs(pi-target_pi))),
                 'min_sample_probability':min(self.mass.values())}
        if max(summary['max_probability_sum_error'],summary['max_FIP_error'])>tolerance:
            raise InvariantError(f'design constraints violated: {summary}')
        return summary

    def exchange(self,a,b,k,l,epsilon):
        a,b=tuple(sorted(a)),tuple(sorted(b))
        exchange_direction(a,b,k,l)
        if a==b or a not in self.mass or b not in self.mass:
            raise ValueError('two distinct active samples required')
        cap=min(self.mass[a],self.mass[b])
        if not np.isfinite(epsilon) or epsilon<0 or epsilon>cap:
            raise ValueError('epsilon outside feasible interval')
        aa=tuple(sorted((set(a)-{k})|{l})); bb=tuple(sorted((set(b)-{l})|{k}))
        changes=defaultdict(float)
        for s,v in [(a,-epsilon),(b,-epsilon),(aa,epsilon),(bb,epsilon)]:
            changes[s]+=v
        for s,v in changes.items():
            p=self.mass.get(s,0.)+v
            if abs(p)<=1e-15:
                self.pruned_mass+=abs(p)
                self.mass.pop(s,None)
            else:
                self.mass[s]=p

    def entropy(self):
        return -fsum(p*np.log(p) for p in self.mass.values() if p>0)
