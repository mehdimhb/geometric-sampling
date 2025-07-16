from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from . import PopulationSimple
from ..structs import Sample
from ..design import Design
from ..measure import Density


class KMeansSpatialSamplingSimple:
    def __init__(
        self,
        coordinate: NDArray,
        inclusion_probability: NDArray,
        *,
        n: int,
        n_zones: int | tuple[int, int],
        tolerance: int,
        split_size: float,
        sort_method: str = "lexico",  # Options: "lexico", "random"
    ) -> None:
        self.coords = coordinate
        self.probs = inclusion_probability
        self.n = n
        self.n_zones = self._pair(n_zones)
        self.tolerance = tolerance
        self.split_size = split_size
        self.sort_method = sort_method

        self.popu = PopulationSimple(
            self.coords,
            self.probs,
            n_clusters=self.n,
            n_zones=n_zones,
            tolerance=self.tolerance,
            split_size=self.split_size,
            sort_method=self.sort_method,
        )
        self.rng = np.random.default_rng()

        self.design = self._build_design()
        self.density = self._build_density()
        self.all_samples, self.all_samples_probs = self._get_all_samples_with_probs()


    def _pair(self, n: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(n, int):
            return (n, n)
        else:
            return n

    def sample(self, n_samples: int):
        samples = np.zeros((n_samples, self.n), dtype=int)
        for i in range(n_samples):
            random_number = self.rng.random()
            zone_index = np.searchsorted(
                np.arange(1, step=round(1 / np.prod(self.n_zones), self.tolerance)),
                random_number,
                side="right",
            )
            for j, cluster in enumerate(self.popu.clusters):
                unit_index = np.searchsorted(
                    cluster.zones[zone_index - 1].units[:, 3].cumsum()
                    + (zone_index - 1)
                    * round(1 / np.prod(self.n_zones), self.tolerance),
                    random_number,
                    side="right",
                )
                if np.prod(self.n_zones) != len(cluster.zones):
                    warning_msg = (
                        f"Warning: Cluster {j} has {len(cluster.zones)} zones, "
                        f"but expected {np.prod(self.n_zones)} based on n_zones={self.n_zones}"
                    )
                    print(warning_msg)
                samples[i, j] = cluster.zones[zone_index - 1].units[unit_index, 0]

        return samples


    def _build_design(self) -> Design:
        # 1. prep
        Z = int(np.prod(self.n_zones))
        zone_width = round(1.0 / Z, self.tolerance)
        design = Design(inclusions=None)

        # 2. collect *all* break‐points: zone boundaries + unit boundaries
        cuts = set()
        # zone boundaries at 0, zone_width, 2*zone_width, …, 1
        for k in range(Z + 1):
            cuts.add(round(k * zone_width, self.tolerance))

        # within each cluster, within each zone, the cumulative unit‐prob cuts
        for cl in self.popu.clusters:
            for zi, zone in enumerate(cl.zones):
                start = round(zi * zone_width, self.tolerance)
                cum = np.cumsum(zone.units[:, 3])
                for c in cum:
                    cuts.add(round(start + c, self.tolerance))

        # 3. sort & clip into [0,1]
        pts = sorted(c for c in cuts if 0.0 <= c <= 1.0)

        # 4. walk the intervals
        last = pts[0]
        for p in pts[1:]:
            length = round(p - last, self.tolerance)
            if length <= 0:
                last = p
                continue

            mid = last + length / 2.0

            # figure out which zone index across *all* clusters
            zone_edges = np.arange(1.0, step=zone_width)
            zone_idx = np.searchsorted(zone_edges, mid, side="right") - 1

            # for that zone, for each cluster find the unit
            ids = []
            for cl in self.popu.clusters:
                z = cl.zones[zone_idx]
                cum_u = z.units[:, 3].cumsum()
                # shift by zone start
                mapped = cum_u + zone_idx * zone_width
                u_idx = np.searchsorted(mapped, mid, side="right")
                ids.append(int(z.units[u_idx, 0]))

            design.push(Sample(length, frozenset(ids)))
            last = p

        design.merge_identical()

        return design

    def _build_density(self) -> Density:
        labels = np.argmax(self.popu.dbk.membership, axis=1)
        centroids = np.vstack([
            self.coords[labels == i].mean(axis=0) for i in range(self.n)
        ])
        return Density(
            self.coords,
            self.probs,
            self.n,
            self.split_size,
            labels,
            centroids
        )

    def _get_all_samples_with_probs(self):
        samples = []
        samples_probs = []
        for sample_obj in self.design:
            samples.append(list(sample_obj.ids))
            samples_probs.append(sample_obj.probability)
        return np.array(samples), np.array(samples_probs)

    @cached_property
    def _get_density_scores(self):
        return self.density.score(self.all_samples)

    def expected_score(self, all_samples_scores=None):
        if all_samples_scores is None:
            all_samples_scores = self._get_density_scores
        return np.sum(all_samples_scores * self.all_samples_probs)

    def var_score(self, all_samples_scores=None):
        if all_samples_scores is None:
            all_samples_scores = self._get_density_scores
        return np.sum(all_samples_scores**2 * self.all_samples_probs) - self.expected_score(all_samples_scores)**2
