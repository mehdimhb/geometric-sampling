import heapq
import itertools
from dataclasses import dataclass
from typing import Generator, Any
import numpy as np
from tqdm import trange

from ..criteria.criteria import Criteria
from ..design import Design

@dataclass(frozen=True, order=False, eq=False)
class Node:
    criteria_value: float
    design: Design

    def __lt__(self, other: Any) -> bool:
        return self.criteria_value < other.criteria_value

class AStar:
    def __init__(
        self,
        inclusions: np.ndarray,
        criteria: Criteria,
        *,
        num_initial_nodes: int = 1,
        threshold_x: float = 1.0,
        threshold_y: float = 1.0,
        switch_coefficient: float = 0.5,
        show_results: int = 0,
        random_pull: bool = False,
        rng: np.random.Generator = np.random.default_rng(),
        initial_design: Design = None,
        initial_design_to_use: int = None,   # NEW parameter!
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

        self.initial_designs = []
        self.initial_nodes = []
        self.initial_info = []

        if initial_design is not None:
            # User provided an initial design
            _ = self.criteria(initial_design)
            var_nht = self.criteria.var_NHT
            var_nht_y = self.criteria.var_NHT_y
            self.initial_designs.append(initial_design)
            self.initial_nodes.append((var_nht, 0, initial_design))
            self.initial_info.append(
                {"var_nht": var_nht, "var_nht_y": var_nht_y, "perm": getattr(initial_design, 'perm', None)}
            )
        else:
            for idx in trange(num_initial_nodes, desc="Generating initial designs"):
                perm = self.rng.permutation(len(inclusions))
                shuffled_inclusions = inclusions[perm]
                perm = [int(v) for v in perm]
                design = Design(inclusions=shuffled_inclusions, perm=perm, rng=rng)
                _ = self.criteria(design)  # Compute criteria on design
                var_nht = self.criteria.var_NHT
                var_nht_y = self.criteria.var_NHT_y
                self.initial_designs.append(design)
                self.initial_nodes.append((var_nht, idx, design))
                self.initial_info.append(
                    {"var_nht": var_nht, "var_nht_y": var_nht_y, "perm": perm}
                )
            print()
            # Sort and keep only the best 'initial_design_to_use' initial designs
            k = initial_design_to_use or 1  # Default to 1 if not specified
            sort_idx = np.argsort([node[0] for node in self.initial_nodes])[:k]
            self.initial_designs = [self.initial_designs[i] for i in sort_idx]
            self.initial_nodes = [self.initial_nodes[i] for i in sort_idx]
            self.initial_info = [self.initial_info[i] for i in sort_idx]

        # Identify best of these for stored info/reference
        initial_best_idx = np.argmin([info["var_nht"] for info in self.initial_info])
        self.initial_design = self.initial_designs[initial_best_idx]
        self.initial_criteria_value = self.initial_info[initial_best_idx]["var_nht"]
        self.initial_var_NHT = self.initial_info[initial_best_idx]["var_nht"]
        self.initial_var_NHT_y = self.initial_info[initial_best_idx]["var_nht_y"]

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

    def neighbors(
        self,
        design: Design,
        num_new_nodes: int,
        num_changes: int,
    ) -> Generator[Design, None, None]:
        for _ in range(num_new_nodes):
            yield self.iterate_design(design, num_changes)

    def run(
        self,
        max_iterations: int,
        num_new_nodes: int,
        max_open_set_size: int,
        num_changes: int,
        show_results: int = 1,
    ):
        initial_efficiency_x = np.round(self.threshold_x / self.initial_var_NHT, 3)
        initial_efficiency_y = np.round(self.threshold_y / self.initial_var_NHT_y, 3)

        closed_set = set()
        open_heap = []
        open_set = set()
        counter = itertools.count()

        # Add all kept initial nodes to open set
        for tup in self.initial_nodes:
            heapq.heappush(open_heap, tup)
            open_set.add(tup[2])  # .design

        self.best_criteria_value = self.initial_criteria_value
        self.best_cost = self.initial_var_NHT
        self.best_cost_y = self.initial_var_NHT_y
        self.best_depth = 0

        for it in range(max_iterations):
            if not open_heap:
                break
            print(f"\rProgress: {it/max_iterations:.1%}", end=" ")

            criteria_value, _, current_design = heapq.heappop(open_heap)
            open_set.discard(current_design)
            if current_design in closed_set:
                continue
            closed_set.add(current_design)

            for new_design in self.neighbors(
                current_design, num_new_nodes, num_changes
            ):
                if new_design in closed_set or new_design in open_set:
                    continue

                _ = self.criteria(new_design)
                new_criteria_value = self.criteria.var_NHT
                new_var_NHT = self.criteria.var_NHT
                new_var_NHT_y = self.criteria.var_NHT_y

                # Only keep a limited open set size
                tie_id = next(counter)
                heapq.heappush(open_heap, (new_criteria_value, tie_id, new_design))
                open_set.add(new_design)
                if len(open_heap) > max_open_set_size:
                    _, _, removed_design = heapq.heappop(open_heap)
                    open_set.discard(removed_design)

                # Only update best if this is better
                if new_criteria_value < self.best_criteria_value:
                    self.best_design = new_design
                    self.best_criteria_value = new_criteria_value
                    self.best_cost = new_var_NHT
                    self.best_cost_y = new_var_NHT_y
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
                            f"  Design Depth:        {getattr(new_design, 'changes', 'NA')}\n"
                            f"  Design Size (|D|):   {len(getattr(new_design, 'heap', []))}\n"
                            f"  Open set size:       {len(open_set)}\n"
                        )
                if self.best_criteria_value < self.threshold_x:
                    print("\nEarly stopping due to threshold!\n")
                    if show_results == 1 and (it % 10 == 0 or it == max_iterations - 1):
                        print(
                            f"\n=== Best Solution Updated at Iteration {it} ===\n"
                            f"  Best Cost (x):       {np.round(self.best_cost, 3)}\n"
                            f"  Best Cost (y):       {np.round(self.best_cost_y, 4)}\n"
                            f"  Criteria Value:      {np.round(self.best_criteria_value, 3)}\n"
                            f"  Efficiency x (0→f):  {initial_efficiency_x} → {np.round(self.threshold_x / self.best_cost, 3)}\n"
                            f"  Efficiency y (0→f):  {initial_efficiency_y} → {np.round(self.threshold_y / self.best_cost_y, 4)}\n"
                            f"  Alpha:               {self.switch_coefficient}\n"
                            f"  Design Depth:        {getattr(new_design, 'changes', 'NA')}\n"
                            f"  Design Size (|D|):   {len(getattr(new_design, 'heap', []))}\n"
                            f"  Open set size:       {len(open_set)}\n"
                        )
                    return it

        return max_iterations