"""Isolated Directed repetitions with dependence diagnostics for the notebook.

Reporting extends the objective's metrics, never its candidate selection or loss.
"""
import os
import numpy as np
from prescribed_sip.core.objective import PairObjective
from prescribed_sip.core.dependence import DependenceError
from prescribed_sip.methods.exchange_search import run_exchange


class ReportedPairObjective(PairObjective):
    report_design_diagnostics = True

    def __init__(self, initial, target, pi, boundary_tol):
        super().__init__(initial.N, np.asarray(target, dtype=np.float64))
        self.dependence = DependenceError(pi, boundary_tol)
        self.initial_rho_error = self.dependence.error(initial.pair_values(self) - self.target)

    def metrics(self):
        metrics = super().metrics()  # Raw D2/RMSE are computed exactly as before.
        rho_error = self.dependence.error(self.residual)
        return {**metrics, 'RMSE_rho': rho_error,
                'Recovery_rho': self.dependence.recovery(rho_error, self.initial_rho_error),
                **self.dependence.counts()}


def directed_repetition(repetition, seed, initial, target, pi, evaluations, batch, seconds,
                        boundary_tol=1e-12, rmse_stop=1e-10, stop_inclusive=False):
    outcome = run_exchange(
        initial=initial.copy(),
        objective=ReportedPairObjective(initial, target, pi, boundary_tol),
        pi=np.asarray(pi, dtype=np.float64), seed=int(seed),
        evaluations=evaluations, seconds=seconds,
        method="GFS-Directed", batch=batch, epsilon="optimal", rmse_stop=rmse_stop,
        stop_inclusive=stop_inclusive,
    )
    outcome.details['worker_pid'] = os.getpid()
    outcome.details['rho_boundary_tol'] = boundary_tol
    return repetition, int(seed), outcome
