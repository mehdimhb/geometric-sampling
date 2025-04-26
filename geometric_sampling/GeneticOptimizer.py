from math import isclose
from typing import List, Optional, Tuple, Set
from geometric_sampling.design import DesignGenetic
from geometric_sampling.structs import  Sample

import numpy as np
import geometric_sampling as gm





class GeneticOptimizer:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def partition_design(self,
                         fip_list: list[float],
                         num_partitions: int
    ) -> tuple[dict[int, list[int]], set[int]]:

        target = sum(fip_list) / num_partitions
        partitions, border_units = {}, set()
        current_partition, cumulative_sum = 0, 0.0
        partitions[current_partition] = []

        for index, fip in enumerate(fip_list):
            if (isclose(cumulative_sum + fip, target, abs_tol=1e-9)
                    or cumulative_sum + fip < target):
                partitions[current_partition].append(index)
                cumulative_sum += fip
            else:
                if not isclose(cumulative_sum, target, abs_tol=1e-9):
                    border_units.add(index)
                    cumulative_sum = fip - (target-cumulative_sum)
                    current_partition += 1
                    partitions[current_partition] = []
                    continue

                current_partition += 1
                partitions[current_partition] = []
                cumulative_sum = fip
                partitions[current_partition].append(index)

        return partitions, border_units


    def _chunk_ids(
            self,
            ids: frozenset[int],
            n_parts: int
    ) -> List[set[int]]:

        lst = sorted(ids)
        base, rem = divmod(len(lst), n_parts)
        chunks, idx = [], 0
        for i in range(n_parts):
            size = base + (1 if i < rem else 0)
            chunks.append(set(lst[idx: idx + size]))
            idx += size
        return chunks

    def combine_n_parents(
            self,
            parents: List[DesignGenetic],
            random_pull: bool = False,
    ) -> DesignGenetic:

        n = len(parents)
        leftovers: List[Optional[Sample]] = [None] * n
        child_design = DesignGenetic(inclusions=None, rng=parents[0].rng)

        while any(leftovers) or any(parent.heap for parent in parents):

            pulled: List[Sample] = []
            for i, par in enumerate(parents):
                if leftovers[i] is not None:
                    r = leftovers[i]
                    leftovers[i] = None
                else:
                    r = par.pull(random_pull)
                pulled.append(r)

            length = min(r.probability for r in pulled)
            if length <= 0:
                break

            all_chunks = [self._chunk_ids(r.ids, n) for r in pulled]
            child_ids: set[int] = set()
            for i in range(n):
                child_ids |= all_chunks[i][i]

            child = Sample(length, frozenset(child_ids),
                                  index=[child_design.step, []])
            child_design.push(child)

            for i, r in enumerate(pulled):
                rem = r.probability - length
                if rem > 0:
                    leftovers[i] = Sample(rem, r.ids, index=[-1, []])
                else:
                    leftovers[i] = None

            child_design.step += 1
            child_design.changes += 1

        return child_design


    def _evaluate(self,
                  inclusion: np.ndarray,
                  x: np.ndarray
    ) -> float:
        """
        Build VarNHT criterion and return a fitness to *maximize*.
        Here we want to MINIMIZE variance → return negative variance.
        """
        # you may want to pass other params in, e.g. y, strata, etc.
        nht = gm.criteria.VarNHT(x, inclusion)
        return - nht.variance()

    def _crossover(self,
                   p1: np.ndarray,
                   p2: np.ndarray,
                   partitions: dict[int, list[int]],
                   border_units: Set[int],
                   cx_rate: float
    ) -> np.ndarray:
        """
        With probability cx_rate use n-parent combine; else clone p1.
        Returns a new inclusion vector (normalized to sum to n).
        """
        if self.rng.random() < cx_rate:
            d1 = DesignGenetic(p1, rng=self.rng)
            d2 = DesignGenetic(p2, rng=self.rng)
            child_design = self.combine_n_parents([d1, d2], random_pull=False)
            # reconstruct inclusion from child_design.heap
            N = p1.size
            incl = np.zeros(N)
            for samp in child_design.heap:
                for i in samp.ids:
                    incl[i] += samp.probability
            # renormalize sum → n
            incl *= p1.sum() / incl.sum()
            return incl
        else:
            return p1.copy()

    def _mutate(self,
                inclusion: np.ndarray,
                mut_rate: float,
                sigma: float = 0.05
    ) -> np.ndarray:
        """
        Gaussian‐perturb each gene with probability mut_rate,
        then renormalize to same total.
        """
        if self.rng.random() > mut_rate:
            return inclusion
        noise = self.rng.normal(0, sigma, size=inclusion.shape)
        new = inclusion + noise
        new = np.clip(new, 1e-8, None)
        new *= inclusion.sum() / new.sum()
        return new

    def _tournament(self,
                    pop: List[np.ndarray],
                    fits: np.ndarray,
                    k: int = 2
    ) -> np.ndarray:
        """
        k‐tournament selection: pick k at random, return the best.
        """
        idx = self.rng.choice(len(pop), size=k, replace=False)
        best = idx[np.argmax(fits[idx])]
        return pop[best].copy()

    def optimize(self,
                 x: np.ndarray,
                 N: int,
                 n: float,
                 pop_size: int = 50,
                 generations: int = 100,
                 cx_rate: float = 0.8,
                 mut_rate: float = 0.3,
                 num_partitions: int = 2,
                 elite_size: int = 2
    ) -> Tuple[np.ndarray, float]:
        """
        The main GA loop. Returns (best_inclusion, best_fitness).
        """
        # 1) partition design once on uniform FIPs (or any fip_list you prefer)
        fip_list = [1/N]*N
        partitions, border_units = self.partition_design(fip_list, num_partitions)

        # 2) init population
        pop: List[np.ndarray] = []
        for _ in range(pop_size):
            incl = self.rng.random(N)
            incl *= n / incl.sum()
            pop.append(incl)

        # 3) evaluate
        fits = np.array([self._evaluate(ind, x) for ind in pop])

        # 4) GA loop
        for gen in range(generations):
            new_pop: List[np.ndarray] = []

            # 4a) elitism
            elite_idx = np.argsort(fits)[-elite_size:]
            for idx in elite_idx:
                new_pop.append(pop[idx].copy())

            # 4b) fill
            while len(new_pop) < pop_size:
                p1 = self._tournament(pop, fits, k=2)
                p2 = self._tournament(pop, fits, k=2)
                child = self._crossover(p1, p2,
                                        partitions, border_units, cx_rate)
                child = self._mutate(child, mut_rate)
                new_pop.append(child)

            pop = new_pop
            fits = np.array([self._evaluate(ind, x) for ind in pop])

            best_var = -fits.max()
            print(f"Gen {gen:3d}  best VarNHT = {best_var:.6e}")

        # return best
        best_idx = np.argmax(fits)
        return pop[best_idx], fits[best_idx]