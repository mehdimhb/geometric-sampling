from dataclasses import dataclass
from abc import abstractmethod, ABC

import numpy as np

from ..design import Design


@dataclass
class Criteria(ABC):
    @abstractmethod
    def evaluate(self, design: Design) -> float: ...

    def __call__(self, design: Design) -> float:
        return self.evaluate(design)


class VarNHT(Criteria):
    def evaluate(self, design: Design) -> float:
        return design.nht_variance


class MoranCriteria(Criteria):
    def evaluate(self, design: Design) -> float:
        expected_value, _ = design.moran
        return expected_value


class DensityDisparityCriteria(Criteria):
    def evaluate(self, design: Design) -> float:
        expected_value, _ = design.density_disparity
        return expected_value


class VoronoiCriteria(Criteria):
    def evaluate(self, design: Design) -> float:
        expected_value, _ = design.voronoi
        return expected_value


class LocalBalanceCriteria(Criteria):
    def evaluate(self, design: Design) -> float:
        expected_value, _ = design.local_balance
        return expected_value


class SIPMatchingCriteria(Criteria):
    """RMSE against a target SIP, using each off-diagonal pair once."""

    def __init__(self, target_sip: np.ndarray):
        target = np.array(target_sip, dtype=float, copy=True)
        if target.ndim != 2 or target.shape[0] != target.shape[1] or len(target) < 2:
            raise ValueError("target_sip must be a square matrix with at least two units")
        if not np.all(np.isfinite(target)) or not np.allclose(target, target.T):
            raise ValueError("target_sip must be finite and symmetric")
        target.setflags(write=False)
        self.target_sip = target
        self._upper = np.triu_indices(len(target), k=1)

    def evaluate(self, design: Design) -> float:
        current = design.sip
        if current.shape != self.target_sip.shape:
            raise ValueError("design and target_sip must have the same population size")
        error = current[self._upper] - self.target_sip[self._upper]
        return float(np.sqrt(np.mean(error ** 2)))
