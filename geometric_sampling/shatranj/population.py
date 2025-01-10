from itertools import pairwise

import numpy as np
from numpy.typing import NDArray


class Population:
    def __init__(
        self,
        coordinate: NDArray,
        inclusion_probability: NDArray,
        *,
        n_of_zones: int|tuple[int, int],
        tolerance: int
    ) -> None:
        self.coords = coordinate
        self.probs = inclusion_probability
        self.n_zones = n_of_zones
        self.tolerance = tolerance

        self.N = self.coords.shape[0]
        self.units = np.concatenate([np.arange(1, self.N+1).reshape(-1, 1), self.coords, self.probs.reshape(-1, 1)], axis=1)

        self.regions = self._generate_regions()

    def _generate_regions(self):
        return self._sweep(self.units[np.argsort(self.units[:, 1])], 0.1)

    def _generate_subregions(self):
        subregions = []
        for region in self.regions:
            subregion, _ = self._sweep(region[np.argsort(region[:, 2])], 0.5)
            subregions.append(subregion)
        return subregions

    def _sweep(self, units: NDArray, threshold: float) -> tuple[list[NDArray], list[int]]:
        boarder_units_remainings, zones_indices = self._generate_boarders_and_indices(units, threshold)
        swept_zones = []
        for indices in pairwise(zones_indices):
            zone, boarder_units_remainings = self._sweep_zone(units, boarder_units_remainings, indices, threshold)
            swept_zones.append(zone)
        return swept_zones

    def _generate_boarders_and_indices(self, units: NDArray, threshold: float):
        thresholds = np.arange(round(np.sum(units[:, 3]), self.tolerance), step=threshold)
        indices = np.append(np.searchsorted(units[:, 3].cumsum(), thresholds, side='right'), units.shape[0]-1)
        boarder_units = {index: units[index][3] for index in np.unique(indices)}
        return boarder_units, indices

    def _sweep_zone(self, units: NDArray, boarder_units_remainings: NDArray, indices: tuple[NDArray, NDArray], threshold: float) -> NDArray:
        zone, start_remainder = self._sweep_boarder_unit(
            np.array([]).reshape(0, 4),
            units[indices[0]],
            boarder_units_remainings[indices[0]],
            threshold
        )
        boarder_units_remainings[indices[0]] = start_remainder

        zone = np.concatenate([zone, units[indices[0]+1:indices[1]]])

        zone, stop_remainder = self._sweep_boarder_unit(
            zone,
            units[indices[1]],
            boarder_units_remainings[indices[1]],
            round(threshold-np.sum(zone[:, 3]), self.tolerance)
        )
        boarder_units_remainings[indices[1]] = stop_remainder

        return zone, boarder_units_remainings

    def _sweep_boarder_unit(self, zone: NDArray, unit: NDArray, probability: float, threshold: float) -> tuple[NDArray, float]:
        if probability < 10**-self.tolerance:
            return zone, 0
        if threshold < 10**-self.tolerance:
            return zone, probability
        if probability < threshold-10**-self.tolerance:
            return np.concatenate([zone, np.append(unit[:3], probability).reshape(1, -1)]), 0
        elif probability > threshold+10**-self.tolerance:
            return np.concatenate([zone, np.append(unit[:3], threshold).reshape(1, -1)]), round(probability-threshold, self.tolerance)
        return np.concatenate([zone, np.append(unit[:3], threshold).reshape(1, -1)]), 0
