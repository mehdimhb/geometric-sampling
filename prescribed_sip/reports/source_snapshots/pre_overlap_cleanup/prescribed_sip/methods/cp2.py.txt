"""Enumerated CP(2) exponential family with exact partition-function calibration.

Bondesson author specification, BANOCOSS2011 Section3(a):
p(s) proportional to exp(sum_{i<j} theta_ij I_i(s)I_j(s)), |s|=n.
This implementation uses our explicit log-partition dual and SciPy L-BFGS,
not a claim to reproduce the inaccessible 2012 paper's iterative/sampling code.
First-order exponential tilts are within CP(2), since I_i = sum_{j!=i} I_iI_j/(n-1).
Enumeration limits applicability; no approximate MCMC moments are called exact.
"""
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp
from prescribed_sip.methods.exact import space_for
from prescribed_sip.core.runtime import Monitor,BudgetStop,Outcome


def calibrate_fip(units,log_mass,pi,tolerance=2e-12,max_iterations=100):
    """Newton calibration in the affine FIP subspace, with a residual line search."""
    X=units.toarray()
    alpha=np.zeros(len(pi)); evaluations=0
    for _ in range(max_iterations):
        z=log_mass+X.T@alpha; p=np.exp(z-logsumexp(z)); evaluations+=1
        current=X@p; error=current-pi
        if np.max(abs(error))<=tolerance: return p,evaluations
        covariance=(X*p)@X.T-np.outer(current,current)
        direction=np.linalg.lstsq(covariance,error,rcond=1e-12)[0]
        before=float(np.max(abs(error)))
        step=1.
        while step>1e-8:
            trial=alpha-step*direction
            tz=log_mass+X.T@trial; tp=np.exp(tz-logsumexp(tz)); evaluations+=1
            if np.max(abs(X@tp-pi))<before:
                alpha=trial; break
            step*=.5
        else: raise RuntimeError('FIP calibration stalled')
    raise RuntimeError('FIP calibration exceeded iteration limit')


def fit_cp2(N,n,pi,objective,evaluations,seconds,initial=None,**kwargs):
    start=perf_counter(); space=space_for(N,n); P=space.target_matrix(objective)
    # Reject deterministic units only by restricting their permissible support.
    allowed=np.ones(len(space.samples),dtype=bool)
    X=space.units.toarray()
    for i in np.flatnonzero(pi<=1e-14): allowed &= X[i]==0
    for i in np.flatnonzero(pi>=1-1e-14): allowed &= X[i]==1
    indices=np.flatnonzero(allowed); U=space.units[:,indices]; Q=P[:,indices]
    first_p,calibration_evals=calibrate_fip(U,np.zeros(len(indices)),pi)
    initial_cp=space.support(np.bincount(indices,weights=first_p,minlength=len(space.samples)))
    monitor=Monitor(objective,initial or initial_cp,pi,evaluations,seconds)
    monitor.started=start
    features=sparse.vstack([U,Q],format='csc'); target=np.r_[pi,objective.target]
    last_theta=np.zeros(len(target)); status='completed'; reason=''
    def dual(theta):
        nonlocal last_theta
        monitor.guard(); monitor.candidates+=1
        z=features.T@theta; logZ=logsumexp(z); p=np.exp(z-logZ)
        last_theta=theta.copy()
        return float(logZ-theta@target),np.asarray(features@p-target)
    result=None
    try:
        result=minimize(dual,last_theta,jac=True,method='L-BFGS-B',
                        options={'gtol':1e-11,'ftol':1e-15,'maxiter':evaluations,'maxls':30})
        theta=result.x
        if not result.success: status='calibration_limit'; reason=result.message
    except BudgetStop as e:
        theta=last_theta; reason=str(e)
        status='timeout' if reason=='time_budget' else 'budget_exhausted'
    # Required exact-FIP final calibration is reported separately; finite numerical
    # dual convergence does not by itself preserve FIP to the validation tolerance.
    correction_start=perf_counter()
    logmass=np.asarray(features.T@theta)
    try:
        p,final_calibration_evals=calibrate_fip(U,logmass,pi)
        final=space.support(np.bincount(indices,weights=p,minlength=len(space.samples)))
        monitor.validate(final); objective.initialize(final); monitor.incumbent(final,objective.metrics())
    except RuntimeError as e:
        final=initial_cp; final_calibration_evals=0; status='numerical_failure'; reason=str(e)
    details={'calibration_implementation':'enumerated log-partition dual L-BFGS + Newton FIP tilt',
             'first_order_calibration_evaluations':calibration_evals,
             'final_FIP_calibration_evaluations':final_calibration_evals,
             'mandatory_finalization_seconds':perf_counter()-correction_start,
             'optimizer_success':bool(result.success) if result is not None else False,
             'exact_partition_function':True}
    return monitor.finish(final,status,reason,details)
