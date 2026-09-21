"""Unique-pair objectives; sparse inputs allocate O(number of target pairs)."""
from itertools import combinations
import numpy as np


class PairObjective:
    def __init__(self,N,target):
        self.N=int(N)
        self.full=not isinstance(target,dict)
        if self.full:
            matrix=np.asarray(target,dtype=np.float64)
            if matrix.shape!=(N,N) or not np.all(np.isfinite(matrix)) or not np.allclose(matrix,matrix.T,atol=1e-12,rtol=0):
                raise ValueError('target must be a finite symmetric N by N matrix')
            self.i,self.j=np.triu_indices(N,1)
            self.target=matrix[self.i,self.j].copy()
            self.lookup=None
        else:
            pairs=sorted(target)
            if not pairs or any(i<0 or j>=N or i>=j for i,j in pairs):
                raise ValueError('sparse target requires valid unique unordered pairs')
            self.i=np.array([i for i,j in pairs],dtype=np.int64)
            self.j=np.array([j for i,j in pairs],dtype=np.int64)
            self.target=np.array([target[p] for p in pairs],dtype=np.float64)
            self.lookup={i*N+j:t for t,(i,j) in enumerate(pairs)}
        if not np.all(np.isfinite(self.target)):
            raise ValueError('nonfinite target')
        self.current=np.zeros_like(self.target)
        self.residual=self.current-self.target
        self.D2=float(self.residual@self.residual)

    def indices(self,pairs):
        if self.full:
            return np.array([i*(2*self.N-i-1)//2+j-i-1 for i,j in pairs],dtype=np.int64)
        return np.array([self.lookup[i*self.N+j] for i,j in pairs if i*self.N+j in self.lookup],dtype=np.int64)

    def sample_indices(self,s):
        return self.indices(combinations(s,2))

    def initialize(self,support):
        self.current=support.pair_values(self)
        self.residual=self.current-self.target
        self.D2=float(self.residual@self.residual)
        return self.metrics()

    def local(self,direction):
        pairs=[p for p in direction if self.full or p[0]*self.N+p[1] in self.lookup]
        idx=self.indices(pairs)
        c=np.array([direction[p] for p in pairs],dtype=np.float64)
        return idx,c

    def step(self,direction,cap,policy,rng,fraction=.75):
        idx,c=self.local(direction)
        A=float(self.residual[idx]@c); B=float(c@c)
        if policy=='optimal':
            epsilon=float(np.clip(-A/B,0,cap)) if B>0 else 0.
        elif policy=='fixed': epsilon=float(fraction*cap)
        elif policy=='random': epsilon=float(rng.uniform(0,cap))
        else: raise ValueError('unknown epsilon policy')
        return epsilon,2*A*epsilon+B*epsilon*epsilon

    def apply(self,direction,epsilon):
        idx,c=self.local(direction)
        change=c*epsilon
        self.D2 += float(2*self.residual[idx]@change+change@change)
        self.current[idx]+=change
        self.residual[idx]+=change
        if self.D2<0 and self.D2>-1e-12:
            self.D2=float(self.residual@self.residual)

    def recompute_error(self,support):
        fresh=support.pair_values(self)
        error=float(np.max(np.abs(fresh-self.current),initial=0))
        # Independently check residual/loss too, without silently repairing drift.
        loss=float((fresh-self.target)@(fresh-self.target))
        if abs(loss-self.D2)>1e-10*max(1.,loss):
            raise RuntimeError('incremental loss drift')
        return error

    def metrics(self):
        loss=float(self.residual@self.residual)
        return {'D2':loss,'RMSE2':float(np.sqrt(loss/(self.N*(self.N-1)/2))) if self.full else float('nan'),
                'weighted_RMSE2':float(np.sqrt(loss/(self.N*(self.N-1)/2))),
                'target_RMSE2':float(np.sqrt(loss/len(self.target))),
                'Dinf':float(np.max(np.abs(self.residual),initial=0))}
