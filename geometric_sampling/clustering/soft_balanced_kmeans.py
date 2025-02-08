import numpy as np
from numpy.typing import NDArray


class SoftBalancedKMeans:
    def __init__(
        self, k: int, tolerance: int = 9, initial_centroids: NDArray = None
    ) -> None:
        self.final_cost = float("inf")
        self.k = k
        self.tolerance = tolerance
        self.data: NDArray = None
        self.centroids = initial_centroids
        self.labels: NDArray = None
        self.fractional_labels: NDArray = None
        self.clusters_sum: NDArray = None
        self.rng = np.random.default_rng()

    def _initiate_centroids(self, data: NDArray) -> None:
        self.centroids = np.zeros((self.k, 2))
        self.centroids[0] = self.rng.choice(data)
        for i in range(1, self.k):
            distances = np.min(
                np.linalg.norm(data.reshape(-1, 1, 2) - self.centroids[:i], axis=2)
                ** 2,
                axis=1,
            )
            self.centroids[i] = self.rng.choice(data, p=distances / np.sum(distances))

    def _assign(self, data: NDArray) -> None:
        self.labels = np.argmin(
            np.linalg.norm(data.reshape(-1, 1, 2) - self.centroids, axis=2) ** 2, axis=1
        )

    def _update_centroids(self, data: NDArray, balance_step: bool = False) -> None:
        if balance_step:
            self.centroids = np.array(
                [
                    np.mean(data[np.nonzero(self.fractional_labels[:, i])[0]], axis=0)
                    for i in range(self.k)
                ]
            )
        else:
            self.centroids = np.array(
                [np.mean(data[self.labels == i], axis=0) for i in range(self.k)]
            )

    def _cost(self, data: NDArray) -> float:
        return sum(
            [
                np.sum(np.linalg.norm(data[self.labels == i] - self.centroids[i]) ** 2)
                for i in range(self.k)
            ]
        )

    def fit(self, data: NDArray) -> None:
        self.data = data
        if self.centroids is None:
            self._initiate_centroids(self.data)
        prev_cost = np.inf
        while True:
            self._assign(self.data)
            self._update_centroids(self.data)
            current_cost = self._cost(self.data)
            if current_cost + 10**-self.tolerance >= prev_cost:
                self.final_cost = current_cost
                break
            prev_cost = current_cost

    def _generate_fractional_labels(self, probs: NDArray):
        fractional_labels = np.zeros((*self.labels.shape, self.k))
        for i in range(self.labels.shape[0]):
            fractional_labels[i, self.labels[i]] = probs[i]
        return fractional_labels

    def _reassignment_cost(
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
                + 10**-self.tolerance
            )
        else:
            return np.inf

    def _get_transfer_records(self, data: NDArray, top_m: int):
        costs = []

        for i in range(data.shape[0]):
            for j in np.nonzero(self.fractional_labels[i])[0]:
                t_min = np.argmin(
                    [self._reassignment_cost(data[i], j, t) for t in range(self.k)]
                )
                cost = self._reassignment_cost(data[i], j, t_min)
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

    def _is_transfer_impossible(self, transfer_records: NDArray) -> bool:
        return transfer_records[0, 0] == np.inf

    def _stop_codition(self, tol) -> bool:
        return np.all(np.abs(self.clusters_sum - 1) < 10**-tol)

    def _expected_num_transfers(self) -> float:
        max_diff_sum = np.max(self.clusters_sum - self.clusters_sum[:, None])
        mean_nonzero_probs = np.mean(
            self.fractional_labels[np.nonzero(self.fractional_labels)]
        )
        return max(int(np.floor(max_diff_sum / (2 * mean_nonzero_probs))), 1)

    def _numerical_stabilizer(self) -> float:
        self.fractional_labels = np.round(self.fractional_labels, self.tolerance)
        self.fractional_labels *= 1 / np.sum(self.fractional_labels, axis=0)
        self.clusters_sum = np.sum(self.fractional_labels, axis=0)

    def balance(self, probs: NDArray) -> None:
        self.fractional_labels = self._generate_fractional_labels(probs)
        self.clusters_sum = np.sum(self.fractional_labels, axis=0)

        while not self._stop_codition(self.tolerance):
            transfer_records = self._get_transfer_records(
                self.data, top_m=self._expected_num_transfers()
            )
            if self._is_transfer_impossible(transfer_records):
                break
            for data_index, from_cluster_index, to_cluster_index in transfer_records:
                self._transfer(data_index, from_cluster_index, to_cluster_index)
                self.clusters_sum = np.sum(self.fractional_labels, axis=0)
            self._update_centroids(self.data, balance_step=True)

        self._numerical_stabilizer()
