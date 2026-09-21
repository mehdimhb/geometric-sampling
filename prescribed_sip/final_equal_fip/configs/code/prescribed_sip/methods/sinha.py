"""Sinha pair-removal mixture, as specified in Bondesson et al. (2014), pp184–185.

Only the nonnegative pair-removal mixture is named Sinha here. No clipping of
material negative coefficients or improvised fitting under that name is done.
"""
from itertools import combinations
from math import comb,fsum
import numpy as np
from prescribed_sip.core.design import Support


def sinha_parameters(N,n,pi,sip):
    if not 2<=n<=N-2: raise ValueError('Sinha requires 2 <= n <= N-2')
    scale=(N-2)*(N-3)/(n*(n-1))
    # Formula immediately preceding Lemma 1, p184 of version of record.
    return {(i,j):scale*sip[(i,j)]-(N-2)/n*(pi[i]+pi[j])+1.
            for i,j in combinations(range(N),2)}


def sinha_moments(N,n,b):
    if any(v < -1e-12 for v in b.values()) or abs(fsum(b.values())-1)>1e-10:
        raise ValueError('invalid pair-removal mixture')
    row=np.zeros(N)
    for (i,j),v in b.items(): row[i]+=v; row[j]+=v
    pi=n/(N-2)*(1-row)
    scale=n*(n-1)/((N-2)*(N-3))
    sip={(i,j):scale*(1-row[i]-row[j]+b.get((i,j),0.)) for i,j in combinations(range(N),2)}
    return pi,sip


def sinha_support(N,n,b):
    sinha_moments(N,n,b)
    denominator=comb(N-2,n)
    mass={}
    for s in combinations(range(N),n):
        omitted=set(range(N))-set(s)
        p=fsum(b.get(pair,0.) for pair in combinations(sorted(omitted),2))/denominator
        if p>0: mass[s]=p
    return Support(N,n,mass)
