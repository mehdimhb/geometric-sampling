import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


class AggregateBalancedKMeans:
    def __init__(
        self, k: int, *, initial_centroids: NDArray = None, tolerance: int = 5
    ) -> None:
        self.k = k
        self.tolerance = tolerance
        self.Y_features = None
        self.X_features = None
        self.weights = None
        self.m = None
        self.N = None
        self.centroids = initial_centroids
        self.labels: NDArray = None
        self.membership: NDArray = None
        self.Ti: NDArray = None
        self.Tij: NDArray = None
        self.goal: float = None
        self.rng = np.random.default_rng()

    def _generate_goal_j(self) -> float:
        return self.X_features.sum(axis=0)/self.k

    def _generate_goal(self) -> float:
        return np.sum(self._generate_goal_j())

    def _generate_membership(self):
        membership = np.zeros((self.N, self.k))
        for j in range(self.N):
            membership[j, self.labels[j]] = 1
        return membership

    def _generate_Tij(self) -> NDArray:
        return np.sum(
            self.X_features[:, :, np.newaxis] * self.membership[:, np.newaxis, :],
            axis=0,
        ).T

    def _generate_Ti(self) -> NDArray:
        return np.sum(self._generate_Tij(), axis=1)

    def _generate_Tij_cost(self):
        return np.sum((self.weights*(self.Tij-self.goal_j))**2)

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
            return score #if score > 0 else np.inf
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

        # print("cost")
        # print(sorted_transfer_records[:5])
        # print()

        return best_cost, top_m_transfer_records

    def _transfer_percent(self, data_index: int, old_cluster: int, new_cluster: int):
        if (
            self.Ti[old_cluster] >= self.goal
            and self.Ti[new_cluster] >= self.goal
        ) or (
            self.Ti[old_cluster] <= self.goal
            and self.Ti[new_cluster] <= self.goal
        ):
            # print('case ++ or --')
            return min(
                self.membership[data_index, old_cluster],
                ((self.Ti[old_cluster] - self.Ti[new_cluster]) / (2*np.sum(self.X_features[data_index])))
            )
        else:
            # print('case +-')
            return min(
                self.membership[data_index, old_cluster],
                ((self.Ti[old_cluster] - self.goal) / np.sum(self.X_features[data_index])),
                ((self.goal - self.Ti[new_cluster]) / np.sum(self.X_features[data_index])),
            )

    def _transfer(self, data_index: int, old_cluster: int, new_cluster: int) -> None:
        transfer_percent = self._transfer_percent(data_index, old_cluster, new_cluster)

        # print(f'transfer {data_index} from {old_cluster} to {new_cluster}')
        # print(f'BEFROE: T_old_cluster={round(self.Ti[old_cluster], 5)} and T_new_cluster={round(self.Ti[new_cluster], 5)}')
        # print(f'transfer percent: {transfer_percent}')

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
        possible_transfers = self.X_features.sum(axis=1)[:, np.newaxis]*self.membership
        mean_nonzero_transfers = np.mean(possible_transfers[np.nonzero(possible_transfers)])
        return max(int(np.floor(max_diff_sum / (2 * mean_nonzero_transfers))), 1)

    def _update_centroids(self) -> None:
        for i in range(self.k):
            self.centroids[i] = np.mean(
                self.Y_features[self.membership[:, i] > 0], axis=0
            )

    def fit(self, Y_features: NDArray, X_features: NDArray, weights: NDArray) -> None:
        self.Y_features = Y_features
        self.X_features = X_features
        self.weights = weights
        self.m = X_features.shape[1]
        self.N = X_features.shape[0]
        self.goal_j = self._generate_goal_j()
        self.goal = self._generate_goal()

        # print('Goal_j:', self.goal_j)
        # print('Goal:', self.goal)
        # print()

        kmeans = KMeans(
            n_clusters=self.k,
            init=self.centroids if self.centroids is not None else "k-means++",
            n_init=10,
            tol=10**-self.tolerance,
        )
        kmeans.fit(self.Y_features)

        self.centroids = kmeans.cluster_centers_
        self.labels = kmeans.labels_
        self.membership = self._generate_membership()
        self.Tij = self._generate_Tij()
        self.Ti = self._generate_Ti()
        self.Tij_cost = self._generate_Tij_cost()
        iter_ = 0

        while not self._stop_codition(self.tolerance) and iter_ < 1000:
            # print("================================================")
            # print("iter:", iter_)
            # print("\nTij", self.Tij)
            # print("\nTij - goal_j", self.Tij - self.goal_j)
            # print("\nTi", self.Ti)
            # print("\nTij_cost", round(self.Tij_cost, 5))
            # print()
            best_cost, transfer_records = self._get_transfer_records(top_m=self._expected_num_transfers())
            if self._no_transfer_possible(best_cost):
                break
            for data_index, old_cluster, new_cluster in transfer_records:
                if self._is_transfer_possible(old_cluster, new_cluster):
                    self._transfer(data_index, old_cluster, new_cluster)
                    self.Tij = self._generate_Tij()
                    self.Ti = self._generate_Ti()
                    self.Tij_cost = self._generate_Tij_cost()
                    # print(f'AFTER: T_old_cluster={round(self.Ti[old_cluster], 5)} and T_new_cluster={round(self.Ti[new_cluster], 5)}')
                    # print()
            self._update_centroids()
            iter_ += 1

    def get_clusters(self) -> NDArray:
        clusters = []

        for i in range(self.k):
            probs = self.membership[:, i] * self.X_features.reshape(-1)
            ids = np.nonzero(probs)[0]
            units = np.concatenate(
                [ids.reshape(-1, 1), self.Y_features[ids], probs[ids].reshape(-1, 1)],
                axis=1,
            )
            clusters.append(units)

        return clusters
