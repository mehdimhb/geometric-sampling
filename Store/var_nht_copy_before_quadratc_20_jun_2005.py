import numpy as np

from ..geometric_sampling.design import Design
from ..geometric_sampling.criteria.criteria import Criteria

from typing import Optional

class VarNHT(Criteria):
    def __call__(self, design: Design) -> float:
        # Calculate estimators
        NHT_estimator = np.array([
            np.sum(self.auxiliary_variable[list(sample.ids)] /
                   self.inclusion_probability[list(sample.ids)])
            for sample in design
        ])
        NHT_estimator_y = np.array([
            np.sum(self.main_variable[list(sample.ids)] /
                   self.inclusion_probability[list(sample.ids)])
            for sample in design
        ])
        samples_probabilities = np.array([sample.probability for sample in design])
        # Calculate diagnostics
        var_NHT = np.sum((NHT_estimator - np.sum(self.auxiliary_variable)) ** 2 * samples_probabilities)
        var_NHT_y = np.sum((NHT_estimator_y - np.sum(self.main_variable)) ** 2 * samples_probabilities)
        # Store in attributes for later access
        self.var_NHT = var_NHT
        self.var_NHT_y = var_NHT_y
        self.NHT_estimator = NHT_estimator
        self.NHT_estimator_y = NHT_estimator_y
        # Optionally store "all" as a dict or namedtuple if you want
        self.last_result = {
            'var_NHT': var_NHT,
            'var_NHT_y': var_NHT_y,
            'NHT_estimator': NHT_estimator,
            'NHT_estimator_y': NHT_estimator_y
        }
        # Return scalar value for optimization (main criterion)
        return var_NHT  # or whichever is the canonical value




        # nht_estimator = np.array(
        #     [
        #         np.sum(
        #             self.auxiliary_variable[list(sample.ids)]
        #             / self.inclusion_probability[list(sample.ids)]
        #         )
        #         for sample in design
        #     ]
        # )

        # samples_probabilities = np.array([sample.probability for sample in design])

        # variance_nht = (
        #     np.sum((nht_estimator**2) * samples_probabilities)
        #     - (np.sum(self.auxiliary_variable)) ** 2
        # )

        # return variance_nht