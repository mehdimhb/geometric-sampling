import numpy as np
from k_means_constrained import KMeansConstrained
from scipy.stats import mode


class DublyBalancedKMeansSimple:
    def __init__(self, k, split_size=0.001):
        self.k = k
        self.split_size = split_size

    def _generate_expanded_coords(self, coords, probs):
        counts = (probs / self.split_size).round().astype(int)
        counts[counts == 0] = 1
        expanded_coords = np.repeat(coords, counts, axis=0)
        expanded_idx = np.repeat(np.arange(self.N), counts)
        return expanded_coords, expanded_idx

    def _generate_labels(self, extended_labels, expanded_idx, coords):
        labels = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            assigned_labels = extended_labels[expanded_idx == i]
            if len(assigned_labels) == 0:
                labels[i] = np.argmin(
                    np.linalg.norm(self.centroids - coords[i], axis=1)
                )
            else:
                labels[i] = mode(assigned_labels, keepdims=True)[0][0]
        return labels

    def fit(self, coords, probs):
        self.N = coords.shape[0]
        expanded_coords, expanded_idx = self._generate_expanded_coords(coords, probs)
        cluster_size = len(expanded_idx) // self.k
        kmeans = KMeansConstrained(
            n_clusters=self.k, size_min=cluster_size, size_max=cluster_size + 1
        )
        labels = kmeans.fit_predict(expanded_coords)
        self.centroids = kmeans.cluster_centers_
        self.labels = self._generate_labels(labels, expanded_idx, coords)

        membership = np.zeros((self.N, self.k))
        for i in range(self.N):
            mask = expanded_idx == i
            assigned_labels = labels[mask]
            for j in range(self.k):
                membership[i, j] = np.sum(assigned_labels == j) / len(assigned_labels)

        # Fix: Ensure no unit has sum(membership)==0
        missing = np.where(membership.sum(axis=1) == 0)[0]
        for i in missing:
            assignment, counts = np.unique(
                labels[expanded_idx == i], return_counts=True
            )
            if len(assignment) > 0:
                membership[i, assignment[np.argmax(counts)]] = 1.0

     
        # Build clusters
        clusters = []
        for j in range(self.k):
            ids = np.where(membership[:, j] > 0)[0]
            membership_weights = membership[ids, j]
            units = np.concatenate(
                [
                    ids.reshape(-1, 1),
                    coords[ids],
                    (probs[ids] * membership_weights).reshape(-1, 1),
                ],
                axis=1,
            )
            clusters.append(units)
        self.clusters = clusters
        self.membership = membership

