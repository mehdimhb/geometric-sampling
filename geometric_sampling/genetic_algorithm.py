import time
import numpy as np
from tqdm import trange
from typing import List, Tuple, Optional

from geometric_sampling import random
from geometric_sampling.GeneticOptimizer import GeneticOptimizer
from geometric_sampling.criteria import VarNHT
from geometric_sampling.design import DesignGenetic


class GeneticAlgorithm:
    def __init__(
        self,
        inclusions: List[float],
        auxiliary_variable: np.ndarray,
        inclusion_probability: np.ndarray,
        population_size: int = 100,
        combination_rate: float = 0.2,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.05,
        n_partitions: int = 2,
        switch_coefficient: float = 0.5,
        max_iters: int = 100,
        random_pull: bool = False,
        rng: np.random.Generator = np.random.default_rng(),
        adaptive_mutation: bool = True,
        min_mutation_rate: float = 0.05,
        max_mutation_rate: float = 0.4,
        early_stopping: int = 10,
    ):
        self.inclusions = inclusions
        self.pop_size = population_size
        self.comb_rate = combination_rate
        self.base_mut_rate = mutation_rate
        self.mut_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.n_parts = n_partitions
        self.switch_coef = switch_coefficient
        self.max_iters = max_iters
        self.random_pull = random_pull
        self.rng = rng
        self.optimizer = GeneticOptimizer()

        self.adaptive_mutation = adaptive_mutation
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.early_stopping = early_stopping
        self.no_improvement_count = 0
        self.best_ever_score = -float("inf")

        self.criterion = VarNHT(
            auxiliary_variable=auxiliary_variable,
            inclusion_probability=inclusion_probability,
        )

        self.partition, self.border = self.optimizer.partition_design(
            fip_list=self.inclusions,
            num_partitions=self.n_parts,
        )

        self.population: List[DesignGenetic] = []
        self.best_history: List[float] = []
        self.mutation_rate_history: List[float] = []

    def _initialize_population(self):
        # generate twice as many candidates by default
        count = self.pop_size * 2
        candidates: list[tuple[float, DesignGenetic]] = []

        for _ in trange(count, desc="Generating initial designs"):
            perm = self.rng.permutation(len(self.inclusions))
            incl_perm = np.array(self.inclusions)[perm]
            perm_list = [int(v) for v in perm]

            d = DesignGenetic(inclusions=incl_perm, rng=self.rng)
            d.perm = perm_list
            # give it a few random iterations to diversify
            for _ in range(self.rng.integers(1, 5)):
                d.iterate(
                    random_pull=self.random_pull,
                    switch_coefficient=self.switch_coef,
                    border_units=self.border,
                    partitions=self.partition,
                )

            score = self.criterion(d)  # lower is better
            candidates.append((score, d))

        # select best pop_size designs (lowest score)
        candidates.sort(key=lambda x: x[0])
        self.population = [d for _, d in candidates[:self.pop_size]]

    def _evaluate(self, pop: List[DesignGenetic]) -> np.ndarray:
        vals = np.array([self.criterion(design) for design in pop])
        return -vals

    def _select_parents(self, scores: np.ndarray, n_parents: int) -> List[int]:
        total = scores.sum()
        # if total <= 0:
        #     probs = np.ones_like(scores) / len(scores)
        # else:
        probs = (1 / scores )/np.sum(1 / scores)
        return list(self.rng.choice(len(scores), size=n_parents, replace=True, p=probs))

    def _adjust_mutation_rate(self, current_best_score: float) -> None:
        if current_best_score > self.best_ever_score:
            self.best_ever_score = current_best_score
            self.no_improvement_count = 0
            self.mut_rate = max(
                self.min_mutation_rate,
                self.base_mut_rate * (0.9 ** min(5, self.no_improvement_count)),
            )
        else:
            self.no_improvement_count += 1
            self.mut_rate = min(
                self.max_mutation_rate,
                self.base_mut_rate * (1.2 ** min(10, self.no_improvement_count)),
            )

    def run(self) -> Tuple[DesignGenetic, int]:
        tmp = time.time()
        self._initialize_population()
        self.no_improvement_count = 0
        self.best_ever_score = -float("inf")
        self.mut_rate = self.base_mut_rate

        for it in range(self.max_iters):
            scores = self._evaluate(self.population)
            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]
            self.best_history.append(best_score)

            if self.adaptive_mutation:
                self._adjust_mutation_rate(best_score)
                self.mutation_rate_history.append(self.mut_rate)\

            if it%10==0 or it == self.max_iters - 1:
                print(
                    f"[Iter {it:3d}](VarNHT = {-best_score:.6f})|mean fitness = {np.mean(scores):.6f} | mutation rate = {self.mut_rate:.4f}"
                )

            if self.early_stopping > 0 and self.no_improvement_count >= self.early_stopping:
                print(f"Early stopping after {it + 1} iterations - no improvement for {self.early_stopping} iterations")
                break

            # elite selection
            n_elites = max(1, int(self.elitism_rate * self.pop_size))
            elite_ids = np.argpartition(scores, -n_elites)[-n_elites:]
            new_pop = [self.population[i].copy() for i in elite_ids]

            # Crossover
            n_combis = int(self.comb_rate * self.pop_size)
            parents_idx = self._select_parents(scores, n_parents=2 * n_combis)

            for i in range(n_combis):
                p1 = self.population[parents_idx[2 * i]]
                p2 = self.population[parents_idx[2 * i + 1]]
                c1, c2 = self.optimizer.combine_n_parents(
                    parents=[p1, p2],
                )
                new_pop.extend([c1, c2])

            seen: set[int] = set()
            unique_pop: List[DesignGenetic] = []
            for design in new_pop:
                h = hash(design.heap)  # uses the heap’s __hash__
                if h not in seen:
                    seen.add(h)
                    unique_pop.append(design)

            # 2) refill to pop_size with fresh random individuals
            while len(unique_pop) < self.pop_size:
                d = DesignGenetic(self.inclusions, rng=self.rng)
                # give it a few random iterations to seed it
                for _ in range(10):
                    d.iterate(
                        random_pull=self.random_pull,
                        switch_coefficient=self.switch_coef,
                        border_units=self.border,
                        partitions=self.partition,
                    )
                unique_pop.append(d)

            self.population = unique_pop


            for design in self.population:
                if design.changes >5:
                    design.merge_identical()
                    design.changes = 0
            if it ==100:
                print(
                    f"Initialization took {time.time() - tmp} seconds, starting optimization."
                )
            #     tmp = time.time()
            # if it%999==0 and it!=0 :
            #     for design in self.population:
            #         design.show()
        #     #
            if it % 999 == 0 and it > 0:
                counter = 0
                for design in self.population:
                    # for heap in design:
                    #     print(heap)
                    design.show()
        #             # counter+=1
        #             # if counter >:
        #             #     break
        #     #    # Debu
        # #    g plotting disabled for performance
        # #    pass

        final_scores = self._evaluate(self.population)
        best_final = int(np.argmax(final_scores))
        return self.population[best_final], it + 1

    def plot_progress(self, figsize=(12, 6)):
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # fitness
            iterations = range(1, len(self.best_history) + 1)
            ax1.plot(iterations, self.best_history, "b-")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Best Fitness")
            ax1.set_title("Optimization Progress")
            ax1.grid(True)

            # mutation
            if self.adaptive_mutation and self.mutation_rate_history:
                ax2.plot(iterations, self.mutation_rate_history, "r-")
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Mutation Rate")
                ax2.set_title("Adaptive Mutation Rate")
                ax2.grid(True)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib is required for plotting progress")
