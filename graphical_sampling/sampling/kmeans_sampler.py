from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Union, Optional

# Import the new builder classes and entities from their expected relative paths
from ..structs import Sample
from ..population import Population
from ..clustering import AuxiliaryBalancedKMeans
from .entity import Zone, Cluster  # Assuming these are in .entity
from .builder import ClusteringZoneBuilder, SweepingZoneBuilder, ClusterBuilder, \
    BaseZoneBuilder  # Assuming these are in .builder
from .order import Order, LexicoXY, LexicoYX, Random, Angle, DistFromOrigin, Projection, DistFromCentroid, \
    Spiral, MaxCoord, Snake, HilbertCurve, Change  # All OrderStrategy implementations
from ..design import Design
from ..measure import Density, Moran, LocalBalance


class KMeansSampler:
    def __init__(
            self,
            population: Population,  # Now takes a Population object directly
            *,
            n: int,  # This n is likely n_clusters for Sampler
            n_zones: int | Tuple[int, int],  # Can be int for clustering, tuple for sweep
            split_size: float,
            zone_builder: str = "sweep",
            units_order: str = "lexico-xy",  # Default for units within zones
            zones_order: str = "lexico-xy",  # Default for zones within clusters
            clusters: List[Cluster] = None,
            labels = None,
            centroids = None,
    ) -> None:
        if not isinstance(population, Population):
            raise TypeError("Input 'population' must be an instance of the Population class.")

        # Store the Population object
        self.population = population
        self.coords = self.population.coords
        self.probs = self.population.probs

        self.n = n  # Number of clusters
        self.split_size = split_size
        self.zone_builder_str = zone_builder

        # Store the current sorting strategies (as strings)
        self._current_units_order_str = units_order
        self._current_zones_order_str = zones_order

        # 1. Convert string sort methods to OrderStrategy instances for initial build
        units_order_strategy = self._get_order_strategy(self._current_units_order_str)
        zones_order_strategy = self._get_order_strategy(self._current_zones_order_str)

        # 2. Initialize ZoneBuilder based on zone_mode
        zone_builder_obj: BaseZoneBuilder
        if self.zone_builder_str == "cluster":
            if not isinstance(n_zones, int):
                raise ValueError("n_zones must be an integer for 'cluster' zone_mode.")
            self.n_zones_value = n_zones  # Store the actual value for later use
            zone_builder_obj = ClusteringZoneBuilder(
                n_zones=self.n_zones_value,
                split_size=self.split_size
            )
        elif self.zone_builder_str == "sweep":
            if isinstance(n_zones, int):
                self.n_zones_value = (n_zones, n_zones)  # Convert int to tuple (n, n) for sweep
            elif isinstance(n_zones, tuple) and len(n_zones) == 2:
                self.n_zones_value = n_zones
            else:
                raise ValueError("n_zones must be an integer or tuple (rows, cols) for 'sweep' zone_mode.")
            zone_builder_obj = SweepingZoneBuilder(
                n_zones=self.n_zones_value
            )
        else:
            raise ValueError(f"Invalid zone_mode: {self.zone_builder_str}. Must be 'cluster' or 'sweep'.")

        # 3. Initialize ClusterBuilder
        self.cluster_builder = ClusterBuilder(
            n_clusters=self.n,
            zone_builder=zone_builder_obj,
            split_size=self.split_size
        )

        # 4. Build clusters (which internally build zones)
        if clusters is None:
            self.clusters, self.labels, self.centroids = self.cluster_builder.build_clusters(self.population)
            self.reorder()
        else:
            self.clusters = clusters
            self.labels = labels
            self.centroids = centroids

        self.rng = np.random.default_rng()

        # Eagerly compute cached properties on first access (handled by @cached_property)
        # We trigger access here to ensure they are available immediately after init.
        _ = self.design
        _ = self.all_samples
        _ = self.all_samples_probs
        _ = self.density

        # Ensure total probability sums to 1 after potential filtering and initial build
        # if np.sum(self.all_samples_probs) > 0:
        #     self.all_samples_probs *= 1 / np.sum(self.all_samples_probs)

    def _get_order_strategy(self, method_name: str) -> Order:
        """Returns an instance of an OrderStrategy based on the method name."""
        strategy_map = {
            "lexico-yx": LexicoYX(),
            "lexico-xy": LexicoXY(),
            "random": Random(),
            "angle_0": Angle(),
            "distance_0": DistFromOrigin(),
            "projection": Projection(),
            "center": DistFromCentroid(),
            "spiral": Spiral(),
            "max": MaxCoord(),
            "snake": Snake(),
            "hilbert": HilbertCurve(min_coord=np.array([0.,0.]), max_coord=np.array([1.,1.])),
            "change": Change(1),
        }

        strategy = strategy_map.get(method_name.lower())
        if strategy is None:
            raise ValueError(f"Unknown sort method: {method_name}. Available: {list(strategy_map.keys())}")
        return strategy

    def reorder(
            self,
            new_units_order: Optional[str] = None,
            new_zones_order: Optional[str] = None
    ) -> None:
        """
        Reorders the units within zones, zones within clusters, and optionally clusters themselves.
        Invalidates and recomputes the sampling design after reordering.

        Args:
            new_units_order (str, optional): The new sorting method for units within zones.
                                                  If None, retains current method.
            new_zones_order (str, optional): The new sorting method for zones within clusters.
                                                   If None, retains current method.
        """
        units_order_to_apply = self._get_order_strategy(
            new_units_order) if new_units_order else self._get_order_strategy(
            self._current_units_order_str)
        zones_order_to_apply = self._get_order_strategy(
            new_zones_order) if new_zones_order else self._get_order_strategy(
            self._current_zones_order_str)

        # Update current method strings
        if new_units_order: self._current_units_order_str = new_units_order
        if new_zones_order: self._current_zones_order_str = new_zones_order

        # 1. Reorder units within zones and zones within clusters
        for cluster in self.clusters:
            # Reorder units within each zone
            for zone in cluster.zones:
                zone.apply_order(units_order_to_apply)

            # Reorder zones within the cluster
            cluster.apply_order(zones_order_to_apply)

        # 3. Invalidate cached properties to force re-computation
        if 'design' in self.__dict__: del self.design
        if 'all_samples' in self.__dict__: del self.all_samples
        if 'all_samples_probs' in self.__dict__: del self.all_samples_probs
        if 'density_scores' in self.__dict__: del self.density_scores
        if 'moran_scores' in self.__dict__: del self.moran_scores
        if 'local_balance_scores' in self.__dict__: del self.local_balance_scores


        # Re-compute relevant properties
        _ = self.design
        _ = self.all_samples
        _ = self.all_samples_probs
        _ = self.density_scores
        _ = self.moran_scores
        _ = self.local_balance_scores

    def reorder_change(
            self,
            n_clusters_to_change_order_zone: int | str = 'all',
            n_clusters_to_change_order_units: int | str = 'all',
            n_zones_to_change_order_units: int | str = 'all',
            n_changes_in_order_of_units: int = 1,
            n_changes_in_order_of_zones: int = 1,
    ) -> None:
        units_order_to_apply = Change(num_changes=n_changes_in_order_of_units)
        zones_order_to_apply = Change(num_changes=n_changes_in_order_of_zones)

        if n_clusters_to_change_order_zone == 'all':
            for cluster in self.clusters:
                cluster.apply_order(zones_order_to_apply)
        else:
            # print('REORDERING ZONES')
            clusters_to_change = np.random.choice(np.arange(len(self.clusters)), size=n_clusters_to_change_order_zone, replace=False)
            # print(clusters_to_change)
            for i in clusters_to_change:
                # print('BEFORE', self.clusters[i])
                self.clusters[i].apply_order(zones_order_to_apply)
                # print('AFTER', self.clusters[i])


        if n_clusters_to_change_order_units == 'all':
            for cluster in self.clusters:
                if n_zones_to_change_order_units == 'all':
                    for zone in cluster.zones:
                        zone.apply_order(units_order_to_apply)
                else:
                    zones_to_change = np.random.choice(np.arange(len(cluster)), size=n_zones_to_change_order_units, replace=False)
                    for i in zones_to_change:
                        cluster.zones[i].apply_order(units_order_to_apply)
        else:
            # print('REORDERING UNITS')
            clusters_to_change = np.random.choice(np.arange(len(self.clusters)), size=n_clusters_to_change_order_units, replace=False)
            # print(clusters_to_change)
            for i in clusters_to_change:
                cluster = self.clusters[i]
                if n_zones_to_change_order_units == 'all':
                    for zone in cluster.zones:
                        zone.apply_order(units_order_to_apply)
                else:
                    zones_to_change = np.random.choice(np.arange(len(cluster)), size=n_zones_to_change_order_units, replace=False)
                    # print(zones_to_change)
                    for j in zones_to_change:
                        # print('BEFORE', cluster.zones[j])
                        # print('BEFORE', cluster.zones[0])
                        cluster.zones[j].apply_order(units_order_to_apply)
                        # print('AFTER', cluster.zones[j])
                        # print('AFTER', cluster.zones[0])


        # 3. Invalidate cached properties to force re-computation
        if 'design' in self.__dict__: del self.design
        if 'all_samples' in self.__dict__: del self.all_samples
        if 'all_samples_probs' in self.__dict__: del self.all_samples_probs
        if 'density_scores' in self.__dict__: del self.density_scores
        if 'moran_scores' in self.__dict__: del self.moran_scores
        if 'local_balance_scores' in self.__dict__: del self.local_balance_scores


        # Re-compute relevant properties
        _ = self.design
        _ = self.all_samples
        _ = self.all_samples_probs
        _ = self.density_scores
        _ = self.moran_scores
        _ = self.local_balance_scores

    def sample(self, n_samples: int):
        if not self.clusters:
            return np.zeros((n_samples, self.n), dtype=int)

        total_conceptual_zones = 0
        if self.zone_builder_str == "cluster":
            total_conceptual_zones = self.n_zones_value
        elif self.zone_builder_str == "sweep":
            total_conceptual_zones = self.n_zones_value[0] * self.n_zones_value[1]

        if total_conceptual_zones == 0:
            total_conceptual_zones = max(1,
                                         len(self.clusters[0].zones) if self.clusters and self.clusters[0].zones else 1)

        # zone_slice_prob = round(1.0 / total_conceptual_zones, 9)

        samples = np.zeros((n_samples, len(self.clusters)), dtype=int)

        for i in range(n_samples):
            random_number = self.rng.random()

            # zone_index = np.searchsorted(
            #     np.arange(zone_slice_prob, 1.0 + 1e-9, zone_slice_prob),
            #     random_number,
            #     side="right",
            # )

            # zone_index = min(zone_index, total_conceptual_zones - 1)
            # zone_index = max(0, zone_index)

            for j, cluster in enumerate(self.clusters):
                if not cluster.zones:
                    samples[i, j] = -1
                    continue

                zone_index = np.searchsorted(
                    cluster.zones_edges,
                    random_number,
                    side="right",
                )

                zone_index -= 1
                zone_index = min(zone_index, total_conceptual_zones - 1)
                zone_index = max(0, zone_index)

                current_zone = cluster.zones[zone_index]

                unit_probs = current_zone.prob

                if unit_probs.size == 0:
                    samples[i, j] = -1
                    continue

                zone_start_global_prob = cluster.zones_edges[zone_index]

                if np.sum(unit_probs) < 1e-9:
                    samples[i, j] = -1
                    continue

                global_cumulative_unit_probs = np.cumsum(unit_probs) + zone_start_global_prob

                unit_selection_idx = np.searchsorted(
                    global_cumulative_unit_probs,
                    random_number,
                    side="right",
                )

                unit_selection_idx = min(unit_selection_idx, len(current_zone) - 1)
                unit_selection_idx = max(0, unit_selection_idx)

                if len(current_zone) > 0:
                    selected_pop_0_based_idx = current_zone.index[unit_selection_idx]
                    samples[i, j] = self.population.ids[selected_pop_0_based_idx]
                else:
                    samples[i, j] = -1

        return samples

    @cached_property
    # @property
    def design(self) -> Design:
        total_conceptual_zones: int
        if self.zone_builder_str == "cluster":
            total_conceptual_zones = self.n_zones_value
        elif self.zone_builder_str == "sweep":
            total_conceptual_zones = self.n_zones_value[0] * self.n_zones_value[1]
        else:
            raise ValueError("Invalid zone_mode for _build_design.")

        if total_conceptual_zones == 0:
            return Design(inclusions=None)

        zone_width = round(1.0 / total_conceptual_zones, 9)
        design = Design(inclusions=None)

        cuts = set()
        # for k in range(total_conceptual_zones + 1):
        #     cuts.add(round(k * zone_width, 9))

        for cluster in self.clusters:
            for edge in cluster.zones_edges:
                cuts.add(edge)

            for zone_idx, zone in enumerate(cluster.zones):
                if len(zone) > 0:
                    cum_unit_probs_within_zone = np.cumsum(zone.prob)
                    for cum_s in cum_unit_probs_within_zone:
                        cuts.add(round(cluster.zones_edges[zone_idx] + cum_s, 9))

        pts = sorted(c for c in cuts if 0.0 <= c <= 1.0)

        if not pts:
            return Design(inclusions=None)
        if pts[0] != 0.0:
            pts.insert(0, 0.0)
        if pts[-1] != 1.0:
            pts.append(1.0)

        last = pts[0]
        for p in pts[1:]:
            length = round(p - last, 9)
            if length <= 0:
                last = p
                continue

            mid = last + length / 2.0

            ids_in_sample = []
            for cluster in self.clusters:

                zone_index = np.searchsorted(
                    cluster.zones_edges,
                    mid,
                    side="right",
                )

                zone_index -= 1
                zone_index = min(zone_index, total_conceptual_zones - 1)
                zone_index = max(0, zone_index)

                current_zone = cluster.zones[zone_index]

                unit_probs = current_zone.prob

                if unit_probs.size == 0:
                    continue

                global_cumulative_unit_shares = np.cumsum(unit_probs) + cluster.zones_edges[zone_index]

                unit_selection_idx = np.searchsorted(
                    global_cumulative_unit_shares,
                    mid,
                    side="right",
                )

                unit_selection_idx = min(unit_selection_idx, len(current_zone) - 1)
                unit_selection_idx = max(0, unit_selection_idx)

                if current_zone.index.size > 0:
                    selected_pop_0_based_idx = current_zone.index[unit_selection_idx]
                    ids_in_sample.append(selected_pop_0_based_idx)

            if ids_in_sample:
                design.push(Sample(length, frozenset(ids_in_sample)))
            last = p

        design.merge_identical()

        return design

    @cached_property
    def density(self) -> Density:
        return Density(
            self.population,
            self.n,
            self.split_size,
            self.labels,
            self.centroids
        )

    @cached_property
    def all_samples(self) -> NDArray:
        samples_list = []
        for sample_obj in self.design:
            if len(sample_obj.ids) == self.n and sample_obj.probability > 1e-9:
                samples_list.append(list(sample_obj.ids))

        if not samples_list:
            return np.array([]).reshape(0, self.n)
        return np.array(samples_list)

    @cached_property
    def all_samples_probs(self) -> NDArray:
        samples_probs_list = []
        for sample_obj in self.design:
            if len(sample_obj.ids) == self.n and sample_obj.probability > 1e-9:
                samples_probs_list.append(sample_obj.probability)

        if not samples_probs_list:
            return np.array([])
        return np.array(samples_probs_list)

    @cached_property
    def fips(self) -> NDArray:
        fip = np.zeros_like(self.probs)
        for sample, prob in zip(self.all_samples, self.all_samples_probs):
            fip[sample] += prob
        return np.array(fip)

    @cached_property
    def density_scores(self):
        if self.all_samples.size == 0:
            return np.array([])
        return self.density.score(self.all_samples)

    @cached_property
    def moran_scores(self):
        if self.all_samples.size == 0:
            return np.array([])
        moran = Moran(self.population)
        return moran.score(self.all_samples)

    @cached_property
    def local_balance_scores(self):
        if self.all_samples.size == 0:
            return np.array([])
        local_balance = LocalBalance(self.population)
        return local_balance.score(self.all_samples)

    def expected_score(self, all_samples_scores: NDArray):
        if all_samples_scores.size == 0 or self.all_samples_probs.size == 0:
            ValueError("There are no samples to compute density scores for.")

        return float(np.sum(all_samples_scores * self.all_samples_probs))

    def expected_density_score(self):
        return self.expected_score(self.density_scores)

    def expected_moran_score(self):
        return self.expected_score(self.moran_scores)

    def expected_local_balance_score(self):
        return self.expected_score(self.local_balance_scores)

    def var_score(self, all_samples_scores: NDArray):
        if all_samples_scores.size == 0 or self.all_samples_probs.size == 0:
            return ValueError("There are no samples to compute density scores for.")

        expected_val = self.expected_score(all_samples_scores)
        return float(np.sum(all_samples_scores ** 2 * self.all_samples_probs) - expected_val ** 2)

    def var_density_score(self):
        return self.var_score(self.density_scores)

    def var_moran_score(self):
        return self.var_score(self.moran_scores)

    def var_local_balance_score(self):
        return self.var_score(self.local_balance_scores)