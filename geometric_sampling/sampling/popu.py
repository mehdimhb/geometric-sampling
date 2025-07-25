from __future__ import annotations

from itertools import pairwise
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError

# optional (used first, then we fall back)
from hilbertcurve.hilbertcurve import HilbertCurve

from .border import Borders
from ..clustering import DoublyBalancedKMeansSimple


# ---------------------------------------------------------------------
# Small Hilbert fallback (pure python)
# ---------------------------------------------------------------------
def _rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y

def hilbert_index_2d(x: int, y: int, p: int) -> int:
    d = 0
    n = 1 << p
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += ((3 * rx) ^ ry) * s * s
        x, y = _rot(n, x, y, rx, ry)
        s >>= 1
    return d

def normalize_to_int_grid(points: NDArray, p: int) -> NDArray:
    pts = np.asarray(points, float)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return np.floor((pts - mn) / (mx - mn + 1e-12) * ((1 << p) - 1)).astype(int)


# ---------------------------------------------------------------------
# Convex-hull helpers (outside class)
# ---------------------------------------------------------------------
def convex_hull_indices(points: NDArray) -> List[int]:
    """Andrew's monotonic chain; returns indices of hull in CCW order."""
    n = len(points)
    if n < 3:
        return list(range(n))

    sorted_idx = np.lexsort((points[:, 1], points[:, 0]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[int] = []
    for idx in sorted_idx:
        while len(lower) >= 2 and cross(points[lower[-2]], points[lower[-1]], points[idx]) <= 0:
            lower.pop()
        lower.append(idx)

    upper: List[int] = []
    for idx in reversed(sorted_idx):
        while len(upper) >= 2 and cross(points[upper[-2]], points[upper[-1]], points[idx]) <= 0:
            upper.pop()
        upper.append(idx)

    return lower[:-1] + upper[:-1]


@dataclass
class Zone:
    units: NDArray  # (m, 4): [id, x, y, p]


@dataclass
class Cluster:
    units: NDArray  # (k, 4)
    zones: List[Zone]


class PopulationSimple:
    def __init__(
        self,
        coordinate: NDArray,
        inclusion_probability: NDArray,
        *,
        n_clusters: int,
        n_zones: Tuple[int, int],
        tolerance: int,
        split_size: float,
        zone_mode: str = "sweep",
        sort_method: str = "lexico-xy",
        zonal_sort: Optional[str] = "lexico-xy",
    ) -> None:
        self.coords = coordinate
        self.probs = inclusion_probability
        self.n_clusters = n_clusters
        self.n_zones = n_zones
        self.tolerance = tolerance
        self.split_size = split_size
        self.zone_mode = zone_mode
        self.sort_method = sort_method
        self.zonal_sort = zonal_sort

        self.dbk: Optional[DoublyBalancedKMeansSimple] = None
        self.clusters: List[Cluster] = self._generate_clusters()
        self.borders = self._generate_borders()

    # ---------------------------- Borders -----------------------------
    def _generate_borders(self) -> Borders:
        border_ids = set(np.where((self.dbk.membership > 0).sum(axis=1) > 1)[0])
        borders = Borders(border_ids)
        borders.build_from(self.clusters)
        return borders

    # --------------------------- Clustering ---------------------------
    def _generate_clusters(self) -> List[Cluster]:
        self.dbk = DoublyBalancedKMeansSimple(k=self.n_clusters, split_size=self.split_size)
        self.dbk.fit(self.coords, self.probs)
        return [
            Cluster(units=units, zones=self._generate_zones(units))
            for units in self.dbk.clusters
        ]

    # ---------------------------- Zones -------------------------------
    def _generate_zones(self, units: NDArray) -> List[Zone]:
        match self.zone_mode:
            case "cluster":
                return self._generate_zones_with_cluster(units)
            case "sweep":
                return self._generate_zones_with_sweep(units)
            case _:
                return [Zone(units=units)]

    def _generate_zones_with_cluster(self, units: NDArray) -> List[Zone]:
        if np.prod(self.n_zones) > 1:
            dbk = DoublyBalancedKMeansSimple(k=np.prod(self.n_zones), split_size=self.split_size)
            dbk.fit(coords=units[:, 1:3], probs=units[:, 3], population_ids=units[:, 0])
            clusters = dbk.clusters
        else:
            clusters = [units]

        zones: List[Zone] = []
        for zone_units in clusters:
            zone_units[:, 3] = self._numerical_stabilizer(zone_units[:, 3])
            idx = self._sort_func(zone_units[:, 1:3], self.sort_method)
            zones.append(Zone(units=zone_units[idx]))
        return self._sort_zones(zones)

    def _generate_zones_with_sweep(self, units: NDArray) -> List[Zone]:
        vertical_zones = self._sweep(units[np.argsort(units[:, 1])], 1 / self.n_zones[0])
        zones: List[Zone] = []
        for zone in vertical_zones:
            basic_zones = self._sweep(zone[np.argsort(zone[:, 2])], 1 / np.prod(self.n_zones))
            for zone_units in basic_zones:
                zone_units[:, 3] = self._numerical_stabilizer(zone_units[:, 3])
                idx = self._sort_func(zone_units[:, 1:3], self.sort_method)
                zones.append(Zone(units=zone_units[idx]))
        return self._sort_zones(zones)

    def _sort_zones(self, zones: List[Zone]) -> List[Zone]:
        if self.zonal_sort is None:
            return zones
        centroids = np.array([z.units[:, 1:3].mean(axis=0) for z in zones])
        centroids = np.nan_to_num(centroids, nan=0.0)
        idx = self._sort_func(self._normalize(centroids), self.zonal_sort)
        return [zones[i] for i in idx]

    # ---------------------------- Sorting -----------------------------
    @staticmethod
    def hilbert_sort_indices(units: NDArray, p: int = 8) -> NDArray:
        ints = normalize_to_int_grid(units, p)
        try:
            h = HilbertCurve(p, 2)
            if hasattr(h, "distance_from_coordinates"):
                d = [h.distance_from_coordinates(list(c)) for c in ints]
            elif hasattr(h, "coordinates_to_distance"):
                d = [h.coordinates_to_distance(list(c)) for c in ints]
            else:
                raise AttributeError
        except Exception:
            d = [hilbert_index_2d(int(x), int(y), p) for x, y in ints]
        return np.argsort(d)

    @staticmethod
    def farthest_point_ordering(units: NDArray, start_idx: Optional[int] = None) -> NDArray:
        n = len(units)
        if start_idx is None:
            c = units.mean(axis=0)
            start_idx = np.argmin(np.linalg.norm(units - c, axis=1))
        selected = [start_idx]
        mind = np.linalg.norm(units - units[start_idx], axis=1)
        for _ in range(1, n):
            mind[selected] = -np.inf
            nxt = np.argmax(mind)
            selected.append(nxt)
            mind = np.minimum(mind, np.linalg.norm(units - units[nxt], axis=1))
        return np.array(selected, dtype=int)

    @staticmethod
    def grid_partition_sort(units: NDArray, grid_size: int = 10) -> NDArray:
        mn = units.min(axis=0)
        mx = units.max(axis=0)
        norm = (units - mn) / (mx - mn + 1e-9)
        bins = np.floor(norm * grid_size).astype(int)
        cell = bins[:, 0] * grid_size + bins[:, 1]
        tie = np.argsort(units[:, 0] + units[:, 1])  # deterministic
        return np.lexsort((tie, cell))

    @staticmethod
    def radial_shell_sort(units: NDArray) -> NDArray:
        c = units.mean(axis=0)
        rel = units - c
        r = np.linalg.norm(rel, axis=1)
        theta = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi)
        return np.lexsort((theta, r))

    @staticmethod
    def spiral_sort_indices(units: NDArray) -> NDArray:
        remaining = list(range(len(units)))
        order: List[int] = []
        while remaining:
            subpts = units[remaining]
            hull_rel = convex_hull_indices(subpts)
            hull_abs = [remaining[i] for i in hull_rel]
            coords_hull = units[hull_abs]
            start_local = np.lexsort((coords_hull[:, 0], coords_hull[:, 1]))[0]
            rotated = hull_abs[start_local:] + hull_abs[:start_local]
            order.extend(rotated)
            for idx in rotated:
                remaining.remove(idx)
        return np.array(order, dtype=int)

    def _sort_func(self, units: NDArray, method: Optional[str]) -> NDArray:
        if method is None:
            return np.arange(units.shape[0], dtype=int)
        match method:
            case "lexico-yx":
                return np.lexsort((units[:, 1], units[:, 0]))
            case "lexico-xy":
                return np.lexsort((units[:, 0], units[:, 1]))
            case "random":
                return np.random.permutation(units.shape[0])
            case "angle_0":
                ang = np.mod(np.arctan2(units[:, 1], units[:, 0]), 2 * np.pi)
                return np.argsort(ang)
            case "distance_0":
                return np.argsort(np.linalg.norm(units, axis=1))
            case "projection":
                return np.argsort(units[:, 0] + units[:, 1])
            case "center":
                c = units.mean(axis=0)
                return np.argsort(np.linalg.norm(units - c, axis=1))
            case "spiral":
                return self.spiral_sort_indices(units)
            case "max":
                return np.argsort(np.max(units, axis=1))
            case "hilbert":
                return self.hilbert_sort_indices(units)
            case "farthest":
                return self.farthest_point_ordering(units)
            case "grid":
                return self.grid_partition_sort(units)
            case "shell":
                return self.radial_shell_sort(units)
            case _:
                return np.arange(units.shape[0], dtype=int)

    # ---------------------------- Sweep -------------------------------
    def _sweep(self, units: NDArray, threshold: float) -> List[NDArray]:
        border_rem, idxs = self._generate_boarders_and_indices(units, threshold)
        zones: List[NDArray] = []
        for i0, i1 in pairwise(idxs):
            zone, border_rem = self._sweep_zone(units, border_rem, (i0, i1), threshold)
            zones.append(zone)
        return zones

    def _generate_boarders_and_indices(
        self, units: NDArray, threshold: float
    ) -> tuple[Dict[int, float], NDArray]:
        thresholds = np.arange(threshold, np.sum(units[:, 3]) - threshold / 2, threshold)
        idx = np.concatenate((
            [0],
            np.searchsorted(units[:, 3].cumsum(), thresholds, side="right"),
            [units.shape[0] - 1],
        ))
        border_units = {i: units[i, 3] for i in np.unique(idx)}
        return border_units, idx

    def _sweep_zone(
        self,
        units: NDArray,
        border_rem: Dict[int, float],
        idx_pair: tuple[int, int],
        threshold: float,
    ) -> tuple[NDArray, Dict[int, float]]:
        start_i, stop_i = idx_pair

        zone, start_left = self._sweep_boarder_unit(
            np.empty((0, 4)), units[start_i], border_rem[start_i], threshold
        )
        border_rem[start_i] = start_left

        zone = np.concatenate([zone, units[start_i + 1:stop_i]])

        zone, stop_left = self._sweep_boarder_unit(
            zone, units[stop_i], border_rem[stop_i], threshold - np.sum(zone[:, 3])
        )
        border_rem[stop_i] = stop_left

        return zone, border_rem

    def _sweep_boarder_unit(
        self, zone: NDArray, unit: NDArray, prob: float, thresh: float
    ) -> tuple[NDArray, float]:
        eps = 10.0 ** (-self.tolerance)
        if prob < eps:
            return zone, 0.0
        if thresh < eps:
            return zone, prob
        if prob < thresh - eps:
            return np.concatenate([zone, np.append(unit[:3], prob).reshape(1, -1)]), 0.0
        elif prob > thresh + eps:
            return np.concatenate([zone, np.append(unit[:3], thresh).reshape(1, -1)]), prob - thresh
        return np.concatenate([zone, np.append(unit[:3], thresh).reshape(1, -1)]), 0.0

    # -------------------------- Utilities -----------------------------
    def _numerical_stabilizer(self, probs: NDArray) -> NDArray:
        p = np.round(probs, self.tolerance)
        p *= 1.0 / (np.sum(p) * np.prod(self.n_zones))
        return p

    @staticmethod
    def _normalize(coords: NDArray) -> NDArray:
        return (coords - coords.min(axis=0)) / (np.ptp(coords, axis=0) + 1e-6)

    @staticmethod
    def get_sorted_cluster_indices_by_lexico(centroids: NDArray, jitter: float = 1e-8):
        noise = np.random.uniform(-jitter, jitter, size=centroids.shape)
        perturbed = centroids + noise
        sorted_order = np.lexsort((perturbed[:, 0], perturbed[:, 1]))
        label_to_color = np.zeros_like(sorted_order)
        for color_idx, label in enumerate(sorted_order):
            label_to_color[label] = color_idx
        return label_to_color, centroids[sorted_order], sorted_order

    # ----------------------------- Plot -------------------------------
    def plot(self, ax=None, figsize: Tuple[int, int] = (8, 6), background_gdf=None):
        bardi_colors = [
            "#4CAF50", "#2196F3", "#F44336", "#FFEB3B", "#FF9800",
            "#9C27B0", "#E91E63", "#00BCD4", "#BDBDBD", "#FFD700"
        ]

        def plot_convex_hull(points, ax_, color, alpha=0.33, edge_color="gray", line_width=0.6):
            points = np.asarray(points)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                return ax_, None
            if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
                return ax_, None
            try:
                hull = ConvexHull(points)
                poly = Polygon(points[hull.vertices],
                               closed=True,
                               facecolor=color,
                               alpha=alpha,
                               edgecolor=edge_color,
                               lw=line_width,
                               zorder=1)
                ax_.add_patch(poly)
                return ax_, hull
            except QhullError:
                return ax_, None

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if background_gdf is not None:
            background_gdf.plot(ax=ax, color="white", edgecolor="black", linewidth=1.5, zorder=0)

        n_clusters = len(self.clusters)
        centroids = np.array([
            cl.units[:, 1:3].mean(axis=0) if len(cl.units) else [np.nan, np.nan]
            for cl in self.clusters
        ])

        _, _, sorted_order = self.get_sorted_cluster_indices_by_lexico(centroids)

        for plot_idx, cl_idx in enumerate(sorted_order):
            cluster = self.clusters[cl_idx]
            color = bardi_colors[plot_idx % len(bardi_colors)]
            pts = cluster.units[:, 1:3]

            ax, _ = plot_convex_hull(pts, ax, color, alpha=0.36, edge_color="black", line_width=0.9)

            ax.scatter(
                pts[:, 0], pts[:, 1],
                color=color, edgecolors="none",
                s=cluster.units[:, 3] * 1000,
                alpha=0.88, zorder=2,
            )

            for z_idx, zone in enumerate(cluster.zones):
                zpts = zone.units[:, 1:3]
                ax, hull = plot_convex_hull(zpts, ax, color, alpha=0.17, edge_color="gray", line_width=0.5)
                center = np.mean(zpts if hull is None else zpts[hull.vertices], axis=0)
                ax.text(center[0], center[1], f"{z_idx + 1}",
                        color="black", fontsize=13, alpha=0.22,
                        ha="center", va="center", weight="bold", zorder=4)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal")
        return ax

    def plot_with_samples(self, samples: NDArray, max_cols: int = 4) -> None:
        n_samples = len(samples)
        n_cols = min(max_cols, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

        for i, sample in enumerate(samples):
            ax = axes[i]
            self.plot(ax=ax)
            ax.scatter(
                self.coords[sample][:, 0], self.coords[sample][:, 1],
                color="black", marker="X", alpha=0.8, s=100, zorder=5
            )
            ax.set_title(f"Sample {i + 1}")

        for ax in axes[n_samples:]:
            fig.delaxes(ax)
