from ..clustering import AggregateBalancedKMeans

import numpy as np
from numpy._typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KernelDensity


class Density:
    def __init__(self, coordinates: NDArray, probabilities: NDArray, k: int):
        # self.coords = (coordinates - coordinates.min(axis=0))/np.ptp(coordinates, axis=0)[0]
        self.coords = coordinates
        self.probs = probabilities
        self.kde = self._kde(coordinates)
        self.labels, self.centroids, self.clusters = self._generate_labels_centroids(k)

    def _kde(self, coords: NDArray) -> KernelDensity:
        kde = KernelDensity(kernel="tophat", bandwidth="scott")
        kde.fit(coords)
        return kde

    def scale(self, scores):
        scaled_scores = np.zeros_like(scores)
        limit = np.sin(np.pi/8)/np.sin(np.pi/4)
        for i, score in enumerate(scores):
            if score >= 0:
                scaled_scores[i] = min(score, limit)/limit
            else:
                scaled_scores[i] = max(score, -limit)/limit
        return scaled_scores

    def _density(self, shifted_coords: NDArray) -> float:
        shifted_kde = self._kde(shifted_coords)
        density = np.exp(self.kde.score_samples(self.coords))
        shifted_density = np.exp(shifted_kde.score_samples(shifted_coords))
        spread = np.mean(self.scale((density-shifted_density)/np.sqrt(density**2+shifted_density**2)))
        var = np.mean(1 - (density+shifted_density)/(np.sqrt(2)*np.sqrt(density**2+shifted_density**2)))
        scale_for_var = 1-np.cos(np.pi/8)
        var_scaled = min(var, scale_for_var)/scale_for_var
        measure = [
            spread,
            var_scaled,
            spread + (np.sign(spread) - spread) * var_scaled,
            # spread + (np.sign(spread) - spread) * var_scaled**2
        ]
        return density, shifted_density, measure

    def _generate_labels_centroids(self, k):
        agg = AggregateBalancedKMeans(k=k, tolerance=5)
        agg.fit(self.coords, self.probs.reshape(-1, 1), np.array([1]))
        labels = np.argmax(agg.membership, axis=1)
        centroids = np.array(
            [
                np.mean(self.coords[labels == i], axis=0)
                for i in range(k)
            ]
        )
        return labels, centroids, agg.get_clusters()

    def _assign_samples_to_centroids(self, samples, centroids):
        cost_matrix = np.linalg.norm(samples[:, :, np.newaxis] - centroids, axis=3).transpose(0, 2, 1)
        return np.array([samples[i][linear_sum_assignment(cost_matrix[i])[1]] for i in range(samples.shape[0])])

    def _generate_shifted_coords(self, shifts: NDArray, labels: NDArray) -> NDArray:
        shifted_coords = self.coords.copy()
        for j, shift in enumerate(shifts):
            shifted_coords[labels == j] += shift
        return shifted_coords

    def _score_sample(
        self, sample: NDArray, labels: NDArray, centroids: NDArray
    ) -> float:
        shifts = sample - centroids
        shifted_coords = self._generate_shifted_coords(shifts, labels)
        return self._density(shifted_coords)

    def score(self, samples: NDArray) -> NDArray:
        scores = []
        densities = []
        samples_assigned = self._assign_samples_to_centroids(
                self.coords[samples], self.centroids
            )
        for sample in samples_assigned:
            density, shifted_density, score = self._score_sample(sample, self.labels, self.centroids)
            densities.append([density, shifted_density])
            scores.append(score)
        return self.zippify(scores), densities

    def zippify(self, scores):
        zipped_scores = [[] for _ in range(len(scores[0]))]
        for score in scores:
            for i, den in enumerate(score):
                zipped_scores[i].append(den)
        return zipped_scores
