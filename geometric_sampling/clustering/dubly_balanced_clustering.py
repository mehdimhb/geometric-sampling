import numpy as np
from numpy.typing import NDArray
from k_means_constrained import KMeansConstrained
from scipy.stats import mode


class DublyBalancedKMeans:
    def __init__(self, k, split_size=0.01):
        self.k = k
        self.split_size = split_size

    def _generate_expanded_coords(self, coords, probs):
        counts = (probs / self.split_size).round().astype(int)
        expanded_coords = np.repeat(coords, counts, axis=0)
        expanded_idx = np.repeat(np.arange(self.N), counts)
        return expanded_coords, expanded_idx

    def _generate_labels(self, extended_labels, expanded_idx, coords):
        labels = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            assigned_labels = extended_labels[expanded_idx == i]
            if len(assigned_labels) == 0:
                labels[i] = np.argmin(np.linalg.norm(self.centroids - coords[i], axis=1))
            else:
                labels[i] = mode(assigned_labels, keepdims=True)[0][0]
        return labels

    def fit(self, coords, probs):
        self.N = coords.shape[0]
        expanded_coords, expanded_idx = self._generate_expanded_coords(coords, probs)
        cluster_size = len(expanded_idx) // self.k
        kmeans = KMeansConstrained(
            n_clusters=self.k,
            size_min=cluster_size,
            size_max=cluster_size+1
        )
        labels = kmeans.fit_predict(expanded_coords)
        self.centroids = kmeans.cluster_centers_
        self.labels = self._generate_labels(labels, expanded_idx, coords)

        cb = ContinuesBalancing(self.k)
        cb.fit(coords, probs, self.centroids, self.labels)

        self.centroids = cb.centroids
        self.labels = cb.labels
        self.membership = cb.membership
        self.Ti = cb.Ti
        self.goal = cb.goal
        self.clusters = cb.get_clusters()



class ContinuesBalancing:
    def __init__(
        self, k: int, *, tolerance: int = 3
    ) -> None:
        self.k = k
        self.tolerance = tolerance
        self.Y_features = None
        self.X_feature = None
        self.labels: NDArray = None
        self.membership: NDArray = None
        self.Ti: NDArray = None
        self.goal: float = None

    def _generate_goal(self) -> float:
        return self.X_feature.sum()/self.k

    def _generate_membership(self):
        membership = np.zeros((self.N, self.k))
        for j in range(self.N):
            membership[j, self.labels[j]] = 1
        return membership

    def _generate_Ti(self) -> NDArray:
        return np.sum(
            self.X_feature[:, np.newaxis] * self.membership,
            axis=0,
        )

    def _transfer_score(
        self,
        data_index: int,
        old_cluster: int,
        new_cluster: int,
    ) -> float:
        if (
            self.Ti[old_cluster] - self.Ti[new_cluster]
            > 10**-self.tolerance
        ):
            score = (
                np.linalg.norm(
                    self.Y_features[data_index] - self.centroids[new_cluster]
                )
                ** 2
                - np.linalg.norm(
                    self.Y_features[data_index] - self.centroids[old_cluster]
                )
                ** 2
            ) / (self.Ti[old_cluster] - self.Ti[new_cluster] + 1e-9)
            return score
        else:
            return np.inf

    def _get_transfer_records(self, top_m: int):
        transfer_records = []

        for i in range(self.N):
            for j_old in np.nonzero(self.membership[i])[0]:
                j_new_min = np.argmin(
                    [self._transfer_score(i, j_old, j_new) for j_new in range(self.k)]
                )
                cost = self._transfer_score(i, j_old, j_new_min)
                transfer_records.append((cost, i, j_old, j_new_min))

        transfer_records = np.array(transfer_records)
        sorted_transfer_records = transfer_records[np.argsort(transfer_records[:, 0])]
        best_cost = sorted_transfer_records[0, 0]
        top_m_transfer_records = sorted_transfer_records[:top_m, 1:].astype(int)

        return best_cost, top_m_transfer_records

    def _transfer_percent(self, data_index: int, old_cluster: int, new_cluster: int):
        if (
            self.Ti[old_cluster] >= self.goal
            and self.Ti[new_cluster] >= self.goal
        ) or (
            self.Ti[old_cluster] <= self.goal
            and self.Ti[new_cluster] <= self.goal
        ):
            return min(
                self.membership[data_index, old_cluster],
                ((self.Ti[old_cluster] - self.Ti[new_cluster]) / (2*np.sum(self.X_feature[data_index])))
            )
        else:
            return min(
                self.membership[data_index, old_cluster],
                ((self.Ti[old_cluster] - self.goal) / np.sum(self.X_feature[data_index])),
                ((self.goal - self.Ti[new_cluster]) / np.sum(self.X_feature[data_index])),
            )

    def _transfer(self, data_index: int, old_cluster: int, new_cluster: int) -> None:
        transfer_percent = self._transfer_percent(data_index, old_cluster, new_cluster)
        self.membership[data_index, old_cluster] -= transfer_percent
        self.membership[data_index, new_cluster] += transfer_percent

    def _no_transfer_possible(self, best_cost: float) -> bool:
        return best_cost == np.inf

    def _is_transfer_possible(self, old_cluster: int, new_cluster: int) -> bool:
        return (
            self.Ti[old_cluster] - self.Ti[new_cluster]
            > 10**-self.tolerance
        )

    def _stop_codition(self, tol) -> bool:
        return np.all(np.abs(self.Ti - self.goal) < 10**-tol)

    def _expected_num_transfers(self) -> float:
        max_diff_sum = np.max(self.Ti - self.Ti[:, None])
        possible_transfers = self.X_feature[:, np.newaxis] * self.membership
        mean_nonzero_transfers = np.mean(possible_transfers[np.nonzero(possible_transfers)])
        return max(int(np.floor(max_diff_sum / (2 * mean_nonzero_transfers))), 1)

    def _update_centroids(self) -> None:
        for i in range(self.k):
            self.centroids[i] = np.mean(
                self.Y_features[self.membership[:, i] > 0], axis=0
            )

    def fit(self, Y_features: NDArray, X_feature: NDArray, centroids: NDArray, labels: NDArray, max_iteration: int = 1000) -> None:
        self.N = len(X_feature)
        self.Y_features = Y_features
        self.X_feature = X_feature
        self.centroids = centroids
        self.labels = labels
        self.goal = self._generate_goal()
        self.membership = self._generate_membership()
        self.Ti = self._generate_Ti()
        self._update_centroids()
        iter_ = 0

        while not self._stop_codition(self.tolerance) and iter_ < max_iteration:
            print(f"\nIteration {iter_}")
            print(f"Ti: {self.Ti}")
            print(f"Sum Ti: {np.sum(self.Ti)}")
            best_cost, transfer_records = self._get_transfer_records(top_m=self._expected_num_transfers())
            if self._no_transfer_possible(best_cost):
                break
            for data_index, old_cluster, new_cluster in transfer_records:
                if self._is_transfer_possible(old_cluster, new_cluster):
                    self._transfer(data_index, old_cluster, new_cluster)
                    self.Ti = self._generate_Ti()
            self._update_centroids()
            iter_ += 1

    def get_clusters(self) -> NDArray:
        clusters = []

        for i in range(self.k):
            x = self.membership[:, i] * self.X_feature
            ids = np.nonzero(x)[0]
            units = np.concatenate(
                [ids.reshape(-1, 1), self.Y_features[ids], x[ids].reshape(-1, 1)],
                axis=1,
            )
            clusters.append(units)

        return clusters