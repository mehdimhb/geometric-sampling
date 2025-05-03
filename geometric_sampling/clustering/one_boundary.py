import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class OneBoundaryBalancedKMeans:
    def __init__(
        self, k: int, *, initial_centroids: NDArray = None, tolerance: int = 5
    ) -> None:
        self.k = k
        self.tolerance = tolerance
        self.Y_features = None
        self.x_feature = None
        self.N = None
        self.centroids = initial_centroids
        self.labels: NDArray = None
        self.membership: NDArray = None
        self.Ti: NDArray = None
        self.goal: float = None
        self.rng = np.random.default_rng()

    def _generate_goal(self) -> float:
        return self.x_feature.sum()/self.k

    def _generate_membership(self):
        membership = np.zeros((self.N, self.k))
        for j in range(self.N):
            membership[j, self.labels[j]] = 1
        return membership

    def _generate_Ti(self) -> NDArray:
        return np.sum(
            self.x_feature[:, np.newaxis] * self.membership,
            axis=0,
        ).T

    def _determine_points_to_gain(self) -> dict[int, list[tuple[int, float]]]:
        """
        For each cluster below goal, compute the points (and fractions) needed to fill its deficit,
        then score each cluster by (mean distance of those points to its centroid) / deficit,
        and finally return the gains for the single cluster with the lowest score.
        Restriction: once any fraction of a point is earmarked for a cluster, that fraction is
        'locked' and cannot be reused for another cluster in this round.  If 100% is locked,
        the point is out entirely; if 50% is locked, only the remaining 50% can be used elsewhere.
        """
        deficits = self.goal - self.Ti
        cluster_gains: dict[int, list[tuple[int, float]]] = {}
        cluster_scores: dict[int, float] = {}
        # tracks fraction of each point already locked in this round

        for cluster_idx, deficit in enumerate(deficits):
            if deficit <= 0:
                continue

            # build list of candidates not yet fully in this cluster and not fully locked
            candidates = []
            for j in range(self.N):
                in_cluster = self.membership[j, cluster_idx]
                locked = self.locked_frac.get(j, 0.0)
                # available fraction of the point for this cluster:
                avail_frac = (1 - in_cluster) - locked
                if avail_frac > 0:
                    candidates.append(j)
            if not candidates:
                continue

            # distances to centroid
            dists = np.linalg.norm(self.Y_features[candidates] - self.centroids[cluster_idx], axis=1)
            sorted_idx = np.argsort(dists)
            sorted_candidates = [candidates[i] for i in sorted_idx]
            # print(f"centroid: {self.centroids[cluster_idx]}")
            # print(dists[sorted_idx][:5], sorted_candidates[:5])

            gains: list[tuple[int, float]] = []
            dists_for_gains: list[float] = []
            accum = 0.0

            for pt_idx in sorted_candidates:
                prev_in = self.membership[pt_idx, cluster_idx]
                locked = self.locked_frac.get(pt_idx, 0.0)
                # fraction of whole point still available to move into this cluster
                avail_frac = (1 - prev_in) - locked
                if avail_frac <= 0:
                    continue
                # x_mass available
                avail_mass = self.x_feature[pt_idx] * avail_frac

                if accum + avail_mass <= deficit:
                    # lock entire available fraction
                    frac_move = avail_frac
                    gains.append((pt_idx, frac_move))
                    dists_for_gains.append(
                        np.linalg.norm(self.Y_features[pt_idx] - self.centroids[cluster_idx])
                    )
                    accum += avail_mass
                    self.locked_frac[pt_idx] = locked + frac_move
                else:
                    # we only need part of the available mass
                    needed = deficit - accum
                    frac_of_mass = needed / self.x_feature[pt_idx]
                    # but that is fraction of whole point:
                    frac_move = min(frac_of_mass, avail_frac)
                    if frac_move > 0:
                        gains.append((pt_idx, frac_move))
                        dists_for_gains.append(
                            np.linalg.norm(self.Y_features[pt_idx] - self.centroids[cluster_idx])
                        )
                        self.locked_frac[pt_idx] = locked + frac_move
                    break

            if gains:
                mean_dist = sum(dists_for_gains) / len(dists_for_gains)
                score = mean_dist / deficit
                cluster_gains[cluster_idx] = gains
                cluster_scores[cluster_idx] = score

        if not cluster_scores:
            return {}

        # choose the cluster with lowest score
        best = min(cluster_scores, key=cluster_scores.get)
        return {best: cluster_gains[best]}


    def _assign_points_to_clusters(self, points_to_gain: dict[int, list[tuple[int, float]]]) -> None:
        """
        Assigns the computed gains for the single target cluster:
        - `frac` is the additional fraction of the point to assign to the cluster.
        - New cluster membership = previous membership + frac (capped at 1.0).
        - The remaining mass (1 − new_membership) is redistributed among the other clusters
        in proportion to their previous memberships.
        """
        if not points_to_gain:
            return

        # unpack the single entry
        cluster_idx, gains = next(iter(points_to_gain.items()))
        for pt_idx, add_frac in gains:
            prev = self.membership[pt_idx].copy()
            # print(f"pt_idx: {pt_idx}, prev: {prev}, add_frac: {add_frac}, cluster_idx: {cluster_idx}")
            already_in = prev[cluster_idx]

            # compute new membership for this cluster
            new_cluster_mem = min(already_in + add_frac, 1.0)
            rem_mass = 1.0 - new_cluster_mem

            # prepare redistribution of remaining mass
            prev_except = prev.copy()
            prev_except[cluster_idx] = 0.0
            total_prev_except = prev_except.sum()

            # build the updated membership vector
            new_mem = np.zeros_like(prev)
            new_mem[cluster_idx] = new_cluster_mem

            if total_prev_except > 0 and rem_mass > 0:
                # scale other clusters proportionally
                new_mem += (prev_except / total_prev_except) * rem_mass
            elif rem_mass > 0:
                # if no other membership existed, put remainder back into this cluster
                new_mem[cluster_idx] += rem_mass

            # apply the update
            self.membership[pt_idx] = new_mem

            # print(f"new_mem: {new_mem}")

        # recompute cluster totals
        self.Ti = self._generate_Ti()

    def _update_centroids(self) -> None:
        for i in range(self.k):
            points = self.Y_features[self.membership[:, i] > 0]
            self.centroids[i] = np.mean(points, axis=0)

    def _stop_codition(self) -> bool:
        return np.all(np.abs(self.Ti - self.goal) < 10**-self.tolerance)

    def fit(self, Y_features: NDArray, x_feature: NDArray) -> None:
        self.Y_features = Y_features
        self.x_feature = x_feature
        self.N = x_feature.size
        self.goal = self._generate_goal()

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
        self.Ti = self._generate_Ti()

        iter_ = 0
        self.locked_frac: dict[int, float] = {}

        while not self._stop_codition() and iter_ < 5:
            print("\nIteration:", iter_)
            print("(BEF) Ti:", self.Ti)
            points_to_gain = self._determine_points_to_gain()
            if not points_to_gain:
                break
            self._assign_points_to_clusters(points_to_gain)
            iter_ += 1

            points = np.array(list(points_to_gain.values())[0])
            points_id, points_frac = points[:, 0], points[:, 1]
            points_id = points_id.astype(int)
            points_frac = points_frac.astype(float)

            mem = np.argmax(self.membership, axis=1)
            # boarders = np.where(np.count_nonzero(self.membership, axis=1)>1)[0]
            plt.figure(figsize=(5, 5), dpi=200)
            plt.scatter(*Y_features.T, c=mem)
            # plt.scatter(*Y_features[boarders].T, c='red', s=10)
            plt.scatter(*Y_features[points_id].T, c='red', marker='*', s=10)
            plt.scatter(*self.centroids.T, c='black', marker='x', s=100)

            for i, point_id in enumerate(points_id):
                x, y = Y_features[point_id]
                plt.text(x, y - 0.05, f"{points_frac[i]:.2f}, {point_id}", color='black', fontsize=8, ha='center')  # Adjust y-offset as needed


            print("(AFT) Ti:", self.Ti)
            print("Diff Ti:", np.sum(np.abs(self.Ti - self.goal)))
            print("Points to gain:", points_to_gain)

            self._update_centroids()
