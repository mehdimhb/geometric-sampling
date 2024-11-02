import numpy as np

from ..design import Design
from .criteria import Criteria


class VarNHT(Criteria):
    def __call__(self, design: Design) -> float:
        nht_estimator = np.array(
            [
                np.sum(
                    self.auxiliary_variable[list(sample.ids)]
                    / self.inclusions[list(sample.ids)]
                )
                for sample in design
            ]
        )

        probabilities = np.array([sample.length for sample in design])
        variance_nht = np.sum(
            (nht_estimator - np.sum(self.auxiliary_variable)) ** 2 * probabilities
        )

        return variance_nht
