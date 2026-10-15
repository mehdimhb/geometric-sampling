"""Small diagnostic probes only: no DI scoring, OT solving, sampling or TSP search."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from graphical_sampling.population import Population
from graphical_sampling.clustering import FIPBalancedNMeans
H=Path(__file__).resolve().parent
# Real historical normalization defect, using the supplied fixed EP construction.
rows=[]
coords=np.column_stack([np.arange(144,dtype=float),np.zeros(144)])
for n in [4,8,16,32]:
 pi=n*np.ones(144)/144;pi*=n/pi.sum();pop=Population(coords,pi)
 rows.append(dict(target_n=n,numpy_sum=repr(float(pi.sum())),compensated_sum=repr(math.fsum(pi)),inferred_n=pop.n,wrong_cluster_count=pop.n!=n))
# Capture exactly which centers the public OT dispatcher forwards. The OT solver is not run.
class RecordingOT(FIPBalancedNMeans):
 def _get_labels_centroids_ot_kmeans(self,coords,inclusions,init_centroids=None):
  self.forwarded_centers=np.asarray(init_centroids).copy()
  return np.zeros(len(coords),dtype=int),self.forwarded_centers
x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
pi=np.full(4,.5);supplied=np.array([[0.,1.],[1.,0.]])
cl=RecordingOT(n=2,init_clust_method='ot');cl._get_labels_centroids(x,pi,supplied)
result=dict(sample_size_probes=rows,OT_dispatch=dict(supplied_centers=supplied.tolist(),actually_forwarded_centers=cl.forwarded_centers.tolist(),honors_supplied_centers=bool(np.array_equal(supplied,cl.forwarded_centers))),new_DI_evaluations=0,new_OT_TSP_searches=0)
(H/'confirmed_defects.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
