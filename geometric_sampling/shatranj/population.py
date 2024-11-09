from typing import Literal
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray


class Population:
    def __init__(
        self,
        coordinate: NDArray,
        inclusion_probability: NDArray,
        *,
        sweeping_mode: Literal['vertical', 'horizontal'],
        region_threshold: float,
        subregion_threshold: float,
        num_of_zones: int|tuple[int, int],
        tolerance: int
    ) -> None:
        self.coords = coordinate
        self.probs = inclusion_probability
        self.sweeping_mode = sweeping_mode
        self.region_threshold = region_threshold
        self.num_zones = num_of_zones
        self.tolerance = tolerance

        self.N = self.coords.shape[0]
        self.units = np.concatenate([np.arange(1, self.N+1).reshape(-1, 1), self.coords, self.probs.reshape(-1, 1)], axis=1)

        self.regions, self.boarder_units = self._generate_regions()
        # self.subregions = self._generate_subregions()

    def _generate_regions(self):
        return self._sweep(self.units[np.argsort(self.units[:, 1])], self.region_threshold)

    def _generate_subregions(self):
        subregions = []
        for region in self.regions:
            subregion, _ = self._sweep(region[np.argsort(region[:, 2])], 0.5)
            subregions.append(subregion)
        return subregions

    def _sweep(self, units: NDArray, threshold: float) -> tuple[list[NDArray], list[int]]:
        areas_indices = self._generate_areas_indices(units, threshold)
        print("areas_indices", areas_indices)
        swept_areas = []
        boarder_units_IDs = []
        remainder: NDArray = None
        for k, indices in enumerate(pairwise(areas_indices)):
            area, remainder, border_unit_ID = self._sweep_area(units, indices, remainder, threshold)
            print(remainder)
            swept_areas.append(area)
            if border_unit_ID is not None:
                boarder_units_IDs.append((border_unit_ID, [k, k+1]))
        return swept_areas, boarder_units_IDs

    def _generate_areas_indices(self, units: NDArray, threshold: float):
        thresholds = np.arange(round(np.sum(units[:, 3]), self.tolerance), step=threshold)
        print("thresholds", thresholds)
        return np.append(np.searchsorted(units[:, 3].cumsum(), thresholds, side='right'), [units.shape[0]])

    def _sweep_area(self, units: NDArray, indices: tuple[NDArray, NDArray], remainder: NDArray|None, threshold: float) -> NDArray:
        print("indices", indices)
        start_index, end_index = 0 if indices[0] == 0 else indices[0]+1, indices[1]
        boarder_index = units.shape[0]-1 if indices[1] == units.shape[0] else indices[1]
        area = self._add_unit(units[start_index:end_index], remainder, False)
        print("round(threshold-np.sum(area[:, 3])", threshold, np.sum(area[:, 3]), round(threshold-np.sum(area[:, 3]), self.tolerance))
        if indices[0] != indices[1]:
            border_unit, remainder = self._split_unit(units[boarder_index], round(threshold-np.sum(area[:, 3]), self.tolerance))
        else:
            border_unit, remainder = self._split_unit(units[boarder_index], round(threshold-np.sum(area[:, 3]), self.tolerance))
        area = self._add_unit(area, border_unit)
        return area, remainder, (int(border_unit[0]) if border_unit is not None else None)

    def _add_unit(self, units: NDArray, unit: NDArray|None, at_end: bool = True) -> NDArray:
        print("area", units)
        if unit is None:
            return units
        if at_end:
            print("end", unit)
            return np.concatenate([units, unit.reshape(1, -1)])
        print("beginning", unit)
        return np.concatenate([unit.reshape(1, -1), units])

    def _split_unit(self, unit: NDArray, splitting_value: float) -> tuple[NDArray, NDArray]:
        first_part = np.append(unit[:3], splitting_value) if splitting_value > 10**-self.tolerance else None
        second_part = (np.append(unit[:3], round(unit[3]-splitting_value, self.tolerance))
                       if (unit[3] - splitting_value) > 10**-self.tolerance else None)
        return first_part, second_part
