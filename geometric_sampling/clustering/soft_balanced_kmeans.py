import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


class SoftBalancedKMeans:
    def __init__(
        self, k: int, *, initial_centroids: NDArray = None, tolerance: int = 3
    ) -> None:
        self.k = k
        self.tolerance = tolerance
        self.coords: NDArray = None
        self.centroids = initial_centroids
        self.labels: NDArray = None
        self.fractional_labels: NDArray = None
        self.clusters_sum: NDArray = None
        self.rng = np.random.default_rng()

    def _generate_fractional_labels(self, probs: NDArray):
        fractional_labels = np.zeros((*self.labels.shape, self.k))
        for i in range(self.labels.shape[0]):
            fractional_labels[i, self.labels[i]] = probs[i]
        return fractional_labels

    def _transfer_score(
        self,
        data_point: NDArray,
        current_cluster_indx: float,
        other_cluster_indx: float,
    ) -> float:
        if (
            self.clusters_sum[current_cluster_indx]
            - self.clusters_sum[other_cluster_indx]
            > 10**-self.tolerance
        ):
            return (
                np.linalg.norm(data_point - self.centroids[other_cluster_indx]) ** 2
                - np.linalg.norm(data_point - self.centroids[current_cluster_indx]) ** 2
            ) / (
                self.clusters_sum[current_cluster_indx]
                - self.clusters_sum[other_cluster_indx]
                + 1e-9
            )
        else:
            return np.inf

    def _get_transfer_records(self, data: NDArray, top_m: int):
        costs = []

        for i in range(data.shape[0]):
            for j in np.nonzero(self.fractional_labels[i])[0]:
                t_min = np.argmin(
                    [self._transfer_score(data[i], j, t) for t in range(self.k)]
                )
                cost = self._transfer_score(data[i], j, t_min)
                costs.append((cost, i, j, t_min))

        costs = np.array(costs)

        return costs[np.argsort(costs[:, 0])][:top_m, 1:].astype(int)

    def _transfer(self, data_index: int, from_index: int, to_index: int) -> None:
        if (
            self.clusters_sum[from_index] >= 1 - 10**-self.tolerance
            and self.clusters_sum[to_index] >= 1 - 10**-self.tolerance
        ) or (
            self.clusters_sum[from_index] <= 1 + 10**-self.tolerance
            and self.clusters_sum[to_index] <= 1 + 10**-self.tolerance
        ):
            transfer_prob = min(
                self.fractional_labels[data_index, from_index],
                (self.clusters_sum[from_index] - self.clusters_sum[to_index]) / 2,
            )
        else:
            transfer_prob = min(
                self.fractional_labels[data_index, from_index],
                self.clusters_sum[from_index] - 1,
                1 - self.clusters_sum[to_index],
            )
        self.fractional_labels[data_index, from_index] = (
            self.fractional_labels[data_index, from_index] - transfer_prob
        )
        self.fractional_labels[data_index, to_index] = (
            self.fractional_labels[data_index, to_index] + transfer_prob
        )

    def _no_transfer_possible(self, transfer_records: NDArray) -> bool:
        return transfer_records[0, 0] == np.inf

    def _is_transfer_possible(self, from_cluster: int, to_cluster: int) -> bool:
        return (
            self.clusters_sum[from_cluster] - self.clusters_sum[to_cluster]
            > 10**-self.tolerance
        )

    def _stop_codition(self, tol) -> bool:
        return np.all(np.abs(self.clusters_sum - 1) < 10**-tol)

    def _expected_num_transfers(self) -> float:
        max_diff_sum = np.max(self.clusters_sum - self.clusters_sum[:, None])
        mean_nonzero_probs = np.mean(
            self.fractional_labels[np.nonzero(self.fractional_labels)]
        )
        return max(int(np.floor(max_diff_sum / (2 * mean_nonzero_probs))), 1)

    def _update_centroids(self, data: NDArray) -> None:
        self.centroids = np.array(
            [
                np.mean(data[np.nonzero(self.fractional_labels[:, i])[0]], axis=0)
                for i in range(self.k)
            ]
        )

    def _numerical_stabilizer(self) -> float:
        self.fractional_labels = np.round(self.fractional_labels, self.tolerance)
        self.fractional_labels *= 1 / np.sum(self.fractional_labels, axis=0)
        self.clusters_sum = np.sum(self.fractional_labels, axis=0)

    def fit(self, data: NDArray, probs: NDArray) -> None:
        self.coords = data

        kmeans = KMeans(
            n_clusters=self.k,
            init=self.centroids if self.centroids is not None else "k-means++",
            n_init=1 if self.centroids is not None else 10,
            tol=10**-self.tolerance,
        )
        kmeans.fit(self.coords)

        self.centroids = kmeans.cluster_centers_
        self.labels = kmeans.labels_
        self.fractional_labels = self._generate_fractional_labels(probs)
        self.clusters_sum = np.sum(self.fractional_labels, axis=0)

        while not self._stop_codition(self.tolerance):
            transfer_records = self._get_transfer_records(
                self.coords, top_m=self._expected_num_transfers()
            )
            if self._no_transfer_possible(transfer_records):
                break
            for data_index, from_cluster_index, to_cluster_index in transfer_records:
                if self._is_transfer_possible(from_cluster_index, to_cluster_index):
                    self._transfer(data_index, from_cluster_index, to_cluster_index)
                    self.clusters_sum = np.sum(self.fractional_labels, axis=0)
            self._update_centroids(self.coords)

        self._numerical_stabilizer()

    def get_clusters(self) -> NDArray:
        clusters = []

        for i in range(self.k):
            probs = self.fractional_labels[:, i]
            ids = np.nonzero(probs)[0]
            units = np.concatenate(
                [ids.reshape(-1, 1), self.coords[ids], probs[ids].reshape(-1, 1)],
                axis=1,
            )
            clusters.append(units)

        return clusters
