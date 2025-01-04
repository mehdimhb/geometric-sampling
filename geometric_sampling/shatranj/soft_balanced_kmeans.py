import numpy as np
from numpy.typing import NDArray


class SoftBalancedKMeans:
    def __init__(self, k: int) -> None:
        self.k = k
        self.centroids: NDArray = None
        self.labels: NDArray = None
        self.rng = np.random.default_rng()

    def initiate_centroids(self, data: NDArray) -> None:
        self.centroids = np.zeros((self.k, 2))
        self.centroids[0] = self.rng.choice(data)
        for i in range(1, self.k):
            distances = np.min(np.linalg.norm(data.reshape(-1, 1, 2)-self.centroids[:i], axis=2)**2, axis=1)
            self.centroids[i] = self.rng.choice(data, p=distances/np.sum(distances))

    def assign(self, data: NDArray) -> None:
        self.labels = np.argmin(np.linalg.norm(data.reshape(-1, 1, 2)-self.centroids, axis=2)**2, axis=1)

    def update_centroids(self, data: NDArray) -> None:
        self.centroids = np.array([np.mean(data[self.labels == i], axis=0) for i in range(self.k)])

    def cost(self, data: NDArray) -> float:
        return sum([np.sum(np.linalg.norm(data[self.labels == i]-self.centroids[i])**2) for i in range(self.k)])

    def fit(self, data: NDArray) -> None:
        self.initiate_centroids(data)
        prev_cost = np.inf
        while True:
            self.assign(data)
            self.update_centroids(data)
            current_cost = self.cost(data)
            if current_cost >= prev_cost:
                self.final_cost = current_cost
                break
            prev_cost = current_cost
