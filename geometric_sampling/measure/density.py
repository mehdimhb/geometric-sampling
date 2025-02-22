from ..clustering import SoftBalancedKMeans

import numpy as np
from numpy._typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt

class Density:
    def __init__(self, coordinates: NDArray, probabilities: NDArray):
        self.coords = coordinates
        self.probs = probabilities
        self.kde = self._kde(coordinates)

    def _kde(self, coords: NDArray) -> KernelDensity:
        kde = KernelDensity(kernel="gaussian", bandwidth="scott")
        kde.fit(coords)
        return kde

    def _norm_density(self, shifted_coords: NDArray) -> float:
        shifted_kde = self._kde(shifted_coords)
        log_density = self.kde.score_samples(self.coords)
        log_density_shifted = shifted_kde.score_samples(shifted_coords)
        return np.linalg.norm(np.exp(log_density) - np.exp(log_density_shifted))

    def _generate_labels_centroids(self, sample_coords: NDArray):
        sbk = SoftBalancedKMeans(
            k=sample_coords.shape[0], initial_centroids=sample_coords
        )
        sbk.fit(self.coords, self.probs)
        labels = np.argmax(sbk.fractional_labels, axis=1)
        centroids = np.array(
            [
                np.mean(self.coords[labels == i], axis=0)
                for i in range(sample_coords.shape[0])
            ]
        )
        return labels, centroids

    def _assign_samples_to_centroids(self, sample: NDArray, centroids: NDArray) -> NDArray:
        cost_matrix = np.linalg.norm(sample[:, np.newaxis] - centroids, axis=2)
        return sample[linear_sum_assignment(cost_matrix)[1]]

    def _generate_shifted_coords(self, shifts: NDArray, labels: NDArray) -> NDArray:
        shifted_coords = self.coords.copy()
        for j, shift in enumerate(shifts):
            shifted_coords[labels == j] += shift
        return shifted_coords

    def _max_score(self, labels: NDArray, centroids: NDArray) -> float:
        shifts = centroids.mean(axis=0) - centroids
        shifted_coords = self._generate_shifted_coords(shifts, labels)
        return self._norm_density(shifted_coords)

    def _scale(self, value: float, n: int) -> float:
        return 1 - (1 - value) ** n

    def _score_sample(
        self, sample: NDArray, labels: NDArray, centroids: NDArray
    ) -> float:
        shifts = sample - centroids
        shifted_coords = self._generate_shifted_coords(shifts, labels)
        return self._scale(
            self._norm_density(shifted_coords) / self._max_score(labels, centroids),
            sample.shape[0],
        )

    def score(self, samples: NDArray) -> NDArray:
        scores = np.zeros(samples.shape[0])
        for i, sample in enumerate(samples):
            labels, centroids = self._generate_labels_centroids(self.coords[sample])
            sample_assigned = self._assign_samples_to_centroids(self.coords[sample], centroids)
            scores[i] = self._score_sample(sample_assigned, labels, centroids)
        return scores
