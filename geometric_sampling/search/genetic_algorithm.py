import numpy as np
from typing import List, Tuple

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
        random_pull:bool = False,
        rng: np.random.Generator = np.random.default_rng(),

    ):
        self.inclusions = inclusions
        self.pop_size = population_size
        self.comb_rate = combination_rate
        self.mut_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.n_parts = n_partitions
        self.switch_coef = switch_coefficient
        self.max_iters = max_iters
        self.random_pull = random_pull
        self.rng = rng
        self.optimizer = GeneticOptimizer()


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

    def _initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            d = DesignGenetic(self.inclusions, rng=self.rng)
            for __ in range(self.rng.integers(1, 5)):
                d.iterate(
                    random_pull=self.random_pull,
                    switch_coefficient=self.switch_coef,
                    border_units=self.border,
                    partitions=self.partition,
                )
            self.population.append(d)

    def _evaluate(self, pop: List[DesignGenetic]) -> np.ndarray:
        vals = np.array([ self.criterion(design) for design in pop ])
        return -vals

    def _select_parents(self, scores: np.ndarray, n_parents: int) -> List[int]:
        total = scores.sum()
        if total <= 0:
            probs = np.ones_like(scores) / len(scores)
        else:
            probs = scores / total
        return list(self.rng.choice(len(scores),
                                    size=n_parents,
                                    replace=True,
                                    p=probs))

    def run(self) -> Tuple[DesignGenetic, int]:
        self._initialize_population()

        for it in range(self.max_iters):
            scores = self._evaluate(self.population)
            best_idx   = int(np.argmax(scores))
            best_score = scores[best_idx]
            self.best_history.append(best_score)
            print(f"[Iter {it:3d}] best fitness = {best_score:.6f} (VarNHT = {-best_score:.6f})")

            n_elites = max(1, int(self.elitism_rate * self.pop_size))
            elite_ids = np.argsort(scores)[-n_elites:]
            new_pop = [ self.population[i].copy() for i in elite_ids ]

            # 4) Crossover ⇒ زوج‌های (2*comb_rate*pop_size)، هر زوج ۲ فرزند
            n_combis = int(self.comb_rate * self.pop_size)
            parents_idx = self._select_parents(scores, n_parents=2*n_combis)

            for i in range(n_combis):
                p1 = self.population[parents_idx[2 * i]]
                p2 = self.population[parents_idx[2 * i + 1]]

                c1, c2 = self.optimizer.combine_n_parents(
                    parents=[p1, p2],
                    random_pull=self.random_pull
                )
                new_pop.extend([c1, c2])

            n_mut = int(self.mut_rate * self.pop_size)
            mutate_ids = self.rng.choice(len(new_pop), size=n_mut, replace=False)
            for mid in mutate_ids:
                new_pop[mid].iterate(
                    random_pull=self.random_pull,
                    switch_coefficient=self.switch_coef,
                    partitions=self.partition,
                    border_units= self.border,
                )

            self.population = new_pop[: self.pop_size]

        final_scores = self._evaluate(self.population)
        best_final = int(np.argmax(final_scores))
        return self.population[best_final], it+1
