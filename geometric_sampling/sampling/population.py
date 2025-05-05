from itertools import pairwise
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from ..clustering import DublyBalancedKMeans


@dataclass
class Zone:
    units: NDArray


@dataclass
class Cluster:
    units: NDArray
    zones: list[Zone]


class Population:
    def __init__(
        self,
        coordinate: NDArray,
        inclusion_probability: NDArray,
        *,
        n_clusters: int,
        n_zones: tuple[int, int],
        tolerance: int,
    ) -> None:
        self.coords = coordinate
        self.probs = inclusion_probability
        self.n_clusters = n_clusters
        self.n_zones = n_zones
        self.tolerance = tolerance

        self.clusters = self._generate_clusters()

    def _generate_clusters(self) -> list[Cluster]:
        # kmeans = SoftBalancedKMeans(self.n_clusters, tolerance=self.tolerance)
        # kmeans.fit(self.coords, self.probs)

        # agg = AggregateBalancedKMeans(k=self.n_clusters, tolerance=self.tolerance)
        # agg.fit(self.coords, self.probs.reshape(-1, 1), np.array([1]))

        dbk = DublyBalancedKMeans(k=self.n_clusters)
        dbk.fit(self.coords, self.probs)

        return [
            Cluster(units=units, zones=self._generate_zones(units))
            # for units in agg.get_clusters()
            for units in dbk.clusters
        ]

    def _generate_zones(self, units) -> list[Zone]:
        vertical_zones = self._sweep(
            units[np.argsort(units[:, 1])], 1 / self.n_zones[0]
        )
        zones = []
        for zone in vertical_zones:
            units_of_basic_zones = self._sweep(
                zone[np.argsort(zone[:, 2])], 1 / (np.prod(self.n_zones))
            )
            for units in units_of_basic_zones:
                units[:, 3] = self._numerical_stabilizer(units[:, 3])
                zones.append(Zone(units=units))
        return zones

    def _sweep(
        self, units: NDArray, threshold: float
    ) -> tuple[list[NDArray], list[int]]:
        boarder_units_remainings, zones_indices = self._generate_boarders_and_indices(
            units, threshold
        )
        swept_zones = []
        for indices in pairwise(zones_indices):
            zone, boarder_units_remainings = self._sweep_zone(
                units, boarder_units_remainings, indices, threshold
            )
            swept_zones.append(zone)
        return swept_zones

    def _generate_boarders_and_indices(self, units: NDArray, threshold: float):
        thresholds = np.arange(
            threshold, np.sum(units[:, 3]) - threshold / 2, threshold
        )
        indices = np.concatenate(
            (
                [0],
                np.searchsorted(units[:, 3].cumsum(), thresholds, side="right"),
                [units.shape[0] - 1],
            )
        )
        boarder_units = {index: units[index][3] for index in np.unique(indices)}
        return boarder_units, indices

    def _sweep_zone(
        self,
        units: NDArray,
        boarder_units_remainings: NDArray,
        indices: tuple[NDArray, NDArray],
        threshold: float,
    ) -> NDArray:
        zone, start_remainder = self._sweep_boarder_unit(
            np.array([]).reshape(0, 4),
            units[indices[0]],
            boarder_units_remainings[indices[0]],
            threshold,
        )
        boarder_units_remainings[indices[0]] = start_remainder

        zone = np.concatenate([zone, units[indices[0] + 1 : indices[1]]])

        zone, stop_remainder = self._sweep_boarder_unit(
            zone,
            units[indices[1]],
            boarder_units_remainings[indices[1]],
            threshold - np.sum(zone[:, 3]),
        )
        boarder_units_remainings[indices[1]] = stop_remainder

        return zone, boarder_units_remainings

    def _sweep_boarder_unit(
        self, zone: NDArray, unit: NDArray, probability: float, threshold: float
    ) -> tuple[NDArray, float]:
        if probability < 10**-self.tolerance:
            return zone, 0
        if threshold < 10**-self.tolerance:
            return zone, probability
        if probability < threshold - 10**-self.tolerance:
            return np.concatenate(
                [zone, np.append(unit[:3], probability).reshape(1, -1)]
            ), 0
        elif probability > threshold + 10**-self.tolerance:
            return np.concatenate(
                [zone, np.append(unit[:3], threshold).reshape(1, -1)]
            ), probability - threshold
        return np.concatenate([zone, np.append(unit[:3], threshold).reshape(1, -1)]), 0

    def _numerical_stabilizer(self, probs: NDArray) -> NDArray:
        probs_stabled = np.round(probs, self.tolerance)
        probs_stabled *= 1 / (np.sum(probs_stabled) * np.prod(self.n_zones))
        return probs_stabled

    def plot(self, ax=None, figsize: tuple[int, int] = (8, 6)) -> None:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        def plot_convex_hull(
            points, ax, color, alpha=0.3, edge_color="black", line_width=1.0
        ):
            if len(points) < 3:
                return ax, None
            hull = ConvexHull(points)
            polygon = Polygon(
                points[hull.vertices],
                closed=True,
                facecolor=color,
                alpha=alpha,
                edgecolor=edge_color,
                lw=line_width,
            )
            ax.add_patch(polygon)
            return ax, hull

        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_points = cluster.units[:, 1:3]
            cluster_color = plt.cm.tab10(cluster_idx % 10)
            ax, _ = plot_convex_hull(cluster_points, ax, color=cluster_color, alpha=0.2)
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                color=cluster_color,
                label=f"Cluster {cluster_idx+1}",
                alpha=0.8,
            )

            for zone_idx, zone in enumerate(cluster.zones):
                zone_points = zone.units[:, 1:3]
                zone_color = cluster_color
                ax, hull = plot_convex_hull(
                    zone_points,
                    ax,
                    color=zone_color,
                    alpha=0.4,
                    edge_color="gray",
                    line_width=0.8,
                )

                hull_center = np.mean(
                    zone_points if hull is None else zone_points[hull.vertices], axis=0
                )
                ax.text(
                    hull_center[0],
                    hull_center[1],
                    f"{zone_idx+1}",
                    color="black",
                    fontsize=16,
                    alpha=0.3,
                    ha="center",
                    va="center",
                    weight="bold",
                )
        return ax

    def plot_with_samples(self, samples: NDArray, max_cols: int = 4) -> None:
        n_samples = len(samples)
        n_cols = min(max_cols, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        figsize = (5 * n_cols, 5 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_samples > 1 else [axes]

        for sample_idx, sample in enumerate(samples):
            ax = axes[sample_idx]
            ax = self.plot(ax)
            ax.scatter(
                self.coords[sample][:, 0],
                self.coords[sample][:, 1],
                color="black",
                marker="X",
                alpha=0.8,
                s=200,
                label=f"Sample {sample_idx+1}",
            )
            ax.set_title(f"Sample {sample_idx+1}")

        for ax in axes[n_samples:]:
            fig.delaxes(ax)
