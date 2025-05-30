import heapq
import itertools
import numpy as np
from joblib import Parallel, delayed
from dataclasses import dataclass

from ..criteria.criteria import Criteria
from ..design import Design

@dataclass(frozen=True, order=False, eq=False)
class Node:
    criteria_value: float
    design: Design

    def __lt__(self, other):
        return self.criteria_value < other.criteria_value

class AStar_elit:
    def __init__(
        self,
        inclusions: np.ndarray,
        criteria: Criteria,
        *,
        num_initial_nodes: int = 1,
        elitism_k: int = 3,               # How many elite solutions to always keep
        threshold_x: float = 1.0,
        threshold_y: float = 1.0,
        switch_coefficient: float = 0.5,
        show_results: int = 0,
        random_pull: bool = False,
        threshold: float = 1e-2,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        self.inclusions = inclusions
        self.criteria = criteria
        self.switch_coefficient = switch_coefficient
        self.show_results = show_results
        self.random_pull = random_pull
        self.threshold_x = threshold_x
        self.threshold_y = threshold_y
        self.rng = rng
        self.num_initial_nodes = num_initial_nodes
        self.elitism_k = elitism_k

        # Multiple initial start nodes with different permutations
        self.initial_designs = []
        self.initial_nodes = []
        info = []
        for idx in range(num_initial_nodes):
            perm = self.rng.permutation(len(inclusions))
            shuffled_inclusions = inclusions[perm]
            perm = [int(v) for v in perm]
            design = Design(inclusions=shuffled_inclusions, perm=perm, rng=rng)
            _ = self.criteria(design)
            var_nht = self.criteria.var_NHT
            var_nht_y = self.criteria.var_NHT_y
            self.initial_designs.append(design)
            self.initial_nodes.append((var_nht, idx, design)) # idx = tiebreaker
            info.append({"var_nht": var_nht, "var_nht_y": var_nht_y, "perm": perm})
        # Use best as initial stats
        bi = np.argmin([v['var_nht'] for v in info])
        self.initial_design = self.initial_designs[bi]
        self.initial_criteria_value = info[bi]["var_nht"]
        self.initial_var_NHT = info[bi]["var_nht"]
        self.initial_var_NHT_y = info[bi]["var_nht_y"]
        self.best_design = self.initial_design
        self.best_criteria_value = self.initial_criteria_value
        self.best_cost = self.initial_var_NHT
        self.best_cost_y = self.initial_var_NHT_y
        self.best_depth = -1

    def iterate_design(self, design: Design, num_changes: int) -> Design:
        new_design = design.copy()
        for _ in range(num_changes):
            new_design.iterate(
                random_pull=self.random_pull,
                switch_coefficient=self.switch_coefficient,
            )
        return new_design

    def neighbors(self, design, num_new_nodes, max_num_changes):
        # Randomize the number of changes
        return [
            self.iterate_design(design, self.rng.integers(1, max_num_changes+1))
            for _ in range(num_new_nodes)
        ]

    def local_search(self, design, k=3):
        # Intensification: try k single-change neighbors, pick the best
        best_design = design
        best_value = self.criteria(design)
        for _ in range(k):
            candidate = self.iterate_design(design, 1)
            value = self.criteria(candidate)
            if value < best_value:
                best_design, best_value = candidate, value
        return best_design

    def run(
        self,
        max_iterations: int,
        num_new_nodes: int,
        max_open_set_size: int,
        max_num_changes: int,
        show_results: int = 1,
        local_search_every: int = 10, # every X iterations, do local search on best
        n_jobs: int = -1,  # parallel jobs for neighbor evaluation
    ):
        initial_efficiency_x = np.round(self.threshold_x / self.initial_var_NHT, 3)
        initial_efficiency_y = np.round(self.threshold_y / self.initial_var_NHT_y, 3)

        closed_set = set()
        open_heap = []
        open_set = set()
        counter = itertools.count()
        elite = []

        # Fill open heap with all diverse initial nodes
        for tup in self.initial_nodes:
            heapq.heappush(open_heap, tup)
            open_set.add(tup[2])

        self.best_criteria_value = self.initial_criteria_value
        self.best_cost = self.initial_var_NHT
        self.best_cost_y = self.initial_var_NHT_y
        self.best_depth = 0

        for it in range(max_iterations):
            if not open_heap:
                break
            print(f"\rProgress: {it/max_iterations:.1%}", end=" ")

            # Pop the best node
            criteria_value, _, current_design = heapq.heappop(open_heap)
            open_set.discard(current_design)
            if current_design in closed_set:
                continue
            closed_set.add(current_design)

            # === Parallel Neighbor Evaluation ===
            neighbor_designs = self.neighbors(current_design, num_new_nodes, max_num_changes)
            # Remove those already seen
            neighbor_designs = [d for d in neighbor_designs if d not in closed_set and d not in open_set]
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.criteria)(neighbor) for neighbor in neighbor_designs
            )
            neighbor_values = [self.criteria.var_NHT for neighbor in neighbor_designs]  # update after each .criteria()

            for nd, nv in zip(neighbor_designs, neighbor_values):
                tie = next(counter)
                heapq.heappush(open_heap, (nv, tie, nd))
                open_set.add(nd)
            # === Elitism: Save top-K elite designs (strict bests seen) ===
            all_candidates = elite + [(self.best_criteria_value, 0, self.best_design)]
            for tup in open_heap:
                all_candidates.append((tup[0], tup[1], tup[2]))
            all_candidates = sorted(all_candidates, key=lambda x: x[0])
            elite = all_candidates[:self.elitism_k]    # top K bests

            # Remove worst from open set if too many, but always keep the elite
            while len(open_heap) > max_open_set_size:
                worst = heapq.heappop(open_heap)
                if worst not in elite:
                    open_set.discard(worst[2])

            # === Update best if found ===
            min_neighbor_idx = int(np.argmin(neighbor_values)) if neighbor_values else None
            for nd, nv in zip(neighbor_designs, neighbor_values):
                if nv < self.best_criteria_value:
                    self.best_design = nd
                    self.best_criteria_value = nv
                    self.best_cost = nv
                    self.best_cost_y = self.criteria.var_NHT_y
                    self.best_depth = it
                    if show_results == 1 and (it % 10 == 0 or it == max_iterations - 1):
                        print(
                            f"\n=== Best Solution Updated at Iteration {it} ===\n"
                            f"  Best Cost (x):       {np.round(self.best_cost, 3)}\n"
                            f"  Best Cost (y):       {np.round(self.best_cost_y, 4)}\n"
                            f"  Criteria Value:      {np.round(self.best_criteria_value, 3)}\n"
                            f"  Efficiency x (0→f):  {initial_efficiency_x} → {np.round(self.threshold_x / self.best_cost, 3)}\n"
                            f"  Efficiency y (0→f):  {initial_efficiency_y} → {np.round(self.threshold_y / self.best_cost_y, 4)}\n"
                            f"  Alpha:               {self.switch_coefficient}\n"
                            f"  Design Depth:        {getattr(nd, 'changes', 'NA')}\n"
                            f"  Design Size (|D|):   {len(getattr(nd, 'heap', []))}\n"
                        )

            # === Periodic Local Search ===
            if it > 0 and (it % local_search_every == 0):
                improved_design = self.local_search(self.best_design, k=3)
                improved_value = self.criteria(improved_design)
                if improved_value < self.best_criteria_value:
                    self.best_design = improved_design
                    self.best_criteria_value = improved_value
                    self.best_cost = improved_value
                    self.best_cost_y = self.criteria.var_NHT_y
                    self.best_depth = it
                    print("\n[Local Search] Improved best solution!\n")
                    print(
                    f"\n=== FINAL BEST SOLUTION ===\n"
                    f"  Found at Iteration:   {self.best_depth}\n"
                    f"  Best Cost (x):        {np.round(self.best_cost, 3)}\n"
                    f"  Best Cost (y):        {np.round(self.best_cost_y, 4)}\n"
                    f"  Criteria Value:       {np.round(self.best_criteria_value, 3)}\n"
                    f"  Efficiency x (0→f):   {initial_efficiency_x} → {np.round(self.threshold_x / self.best_cost, 3)}\n"
                    f"  Efficiency y (0→f):   {initial_efficiency_y} → {np.round(self.threshold_y / self.best_cost_y, 4)}\n"
                    f"  Alpha:                {self.switch_coefficient}\n"
                    f"  Design Depth:         {getattr(self.best_design, 'changes', 'NA')}\n"
                    f"  Design Size (|D|):    {len(getattr(self.best_design, 'heap', []))}\n"
                )
            if self.best_criteria_value < self.threshold_x:
                print("\nEarly stopping due to threshold!\n")
                print(
                    f"\n=== FINAL BEST SOLUTION ===\n"
                    f"  Found at Iteration:   {self.best_depth}\n"
                    f"  Best Cost (x):        {np.round(self.best_cost, 3)}\n"
                    f"  Best Cost (y):        {np.round(self.best_cost_y, 4)}\n"
                    f"  Criteria Value:       {np.round(self.best_criteria_value, 3)}\n"
                    f"  Efficiency x (0→f):   {initial_efficiency_x} → {np.round(self.threshold_x / self.best_cost, 3)}\n"
                    f"  Efficiency y (0→f):   {initial_efficiency_y} → {np.round(self.threshold_y / self.best_cost_y, 4)}\n"
                    f"  Alpha:                {self.switch_coefficient}\n"
                    f"  Design Depth:         {getattr(self.best_design, 'changes', 'NA')}\n"
                    f"  Design Size (|D|):    {len(getattr(self.best_design, 'heap', []))}\n"
                )
                return it

        return max_iterations