"""Reuse notebook projection factor; verify against Loonis (2023), Algorithm3.1.

Source: simulations/jupyters/15_simple_gbfs_vs_dsd.ipynb, real_dsd_factor.
The code below retains its rotation construction, with explicit certainty/zero
unit handling and independent checks. DSD is a structured baseline, not a SIP solver.
"""
from itertools import combinations
import numpy as np


def projection_factor(pi):
    pi=np.asarray(pi,dtype=np.float64)
    N=len(pi); n=int(round(pi.sum()))
    if np.any(pi<0) or np.any(pi>1) or abs(pi.sum()-n)>1e-10:
        raise ValueError('invalid projection diagonal')
    certain=np.flatnonzero(pi>=1-1e-14); live=np.flatnonzero((pi>1e-14)&(pi<1-1e-14))
    if len(live)!=N:
        V=np.zeros((N,n)); V[certain,np.arange(len(certain))]=1.
        if len(live): V[np.ix_(live,np.arange(len(certain),n))]=projection_factor(pi[live])
        return V
    cumulative=0.; previous_integer=0; threshold=1
    s=np.zeros(N); crossing=[]
    for k,probability in enumerate(pi):
        before=cumulative; cumulative+=probability
        if cumulative>=threshold-1e-12:
            alpha=threshold-before
            s[k]=np.sqrt(max(0.,(1-probability)/(1-alpha)))
            crossing.append(k); previous_integer=threshold; threshold+=1
        else:
            s[k]=np.sqrt(probability/(previous_integer+1-before))
        s[k]=np.clip(s[k],0,1)
    V=np.zeros((N,n)); V[0,0]=1.
    for column in range(1,n): V[crossing[column-1]+1,column]=1.
    c=np.sqrt(np.maximum(0.,1-s*s))
    for k in range(N-1):
        upper,lower=V[k].copy(),V[k+1].copy()
        V[k]=s[k]*upper-c[k]*lower
        V[k+1]=c[k]*upper+s[k]*lower
    if np.max(abs(V.T@V-np.eye(n)),initial=0)>1e-10 or np.max(abs(np.sum(V*V,axis=1)-pi))>1e-10:
        raise RuntimeError('projection construction failed validation')
    return V


def projection_moments(V,pairs):
    pi=np.sum(V*V,axis=1)
    return {(i,j):float(pi[i]*pi[j]-np.dot(V[i],V[j])**2) for i,j in pairs}
