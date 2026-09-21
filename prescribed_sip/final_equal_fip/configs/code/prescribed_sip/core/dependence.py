"""Fixed-FIP dependence reporting; does not change the raw SIP search objective."""
import numpy as np


class DependenceError:
    def __init__(self, pi, boundary_tol=1e-12):
        pi = np.asarray(pi, dtype=np.float64)
        if not np.isfinite(boundary_tol) or not 0 <= boundary_tol < .5:
            raise ValueError('RHO_BOUNDARY_TOL must be finite and in [0, 0.5).')
        if (pi.ndim != 1 or not np.all(np.isfinite(pi))
                or np.any(pi < -boundary_tol) or np.any(pi > 1 + boundary_tol)):
            raise ValueError('Prescribed FIP must be a finite probability vector.')
        self.i, self.j = np.triu_indices(len(pi), 1)
        interior = (pi > boundary_tol) & (pi < 1 - boundary_tol)
        self.valid = interior[self.i] & interior[self.j]
        variance = np.clip(pi, 0., 1.) * (1. - np.clip(pi, 0., 1.))
        self.denominator = np.sqrt(variance[self.i[self.valid]] * variance[self.j[self.valid]])
        self.included = int(self.valid.sum())
        self.excluded = int(len(self.valid) - self.included)

    def error(self, residual):
        residual = np.asarray(residual, dtype=np.float64)
        if residual.shape != self.valid.shape:
            raise ValueError('SIP residual must contain every i<j pair in original ID order.')
        if self.included == 0:
            return float('nan')
        # pi_i*pi_j cancels between estimated and target rho (same prescribed FIP).
        standardized = residual[self.valid] / self.denominator
        return float(np.sqrt(np.mean(standardized ** 2)))

    def from_matrices(self, estimated, target):
        return self.error((np.asarray(estimated) - np.asarray(target))[self.i, self.j])

    @staticmethod
    def recovery(final, initial):
        if not np.isfinite(initial) or initial == 0 or not np.isfinite(final):
            return float('nan')
        return 100. * (1. - final / initial)

    def counts(self):
        return {'Rho pairs included': self.included, 'Rho pairs excluded': self.excluded}
