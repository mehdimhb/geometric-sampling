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
        x,
        y,
        threshold_x: float = 1.0,
        threshold_y: float = 1.0,
        switch_coefficient: float = 0.5,
        show_results: int = 0,
        random_pull: bool = False,
        rng: np.random.Generator = np.random.default_rng(),
        initial_design: Design = None,
        initial_design_to_use: int = None,
        var_percent_exected: float = 1,
    ) -> None:
        self.inclusions = inclusions
        self.criteria = criteria
        self.switch_coefficient = switch_coefficient
        self.show_results = show_results
        self.random_pull = random_pull
        self.x = x
        self.y = y
        self.threshold_x = threshold_x
        self.threshold_y = threshold_y
        self.rng = rng
        self.num_initial_nodes = num_initial_nodes
        self.var_percent_exected = var_percent_exected
        self.initial_designs = []
        self.initial_nodes = []
        self.initial_info = []

        if initial_design is not None:
            _ = self.criteria(initial_design)
            var_nht = self.criteria.var_NHT
            var_nht_y = self.criteria.var_NHT_y
            self.initial_designs.append(initial_design)
            self.initial_nodes.append((var_nht, 0, initial_design))
            self.initial_info.append(
                {"var_nht": var_nht, "var_nht_y": var_nht_y, "perm": getattr(initial_design, 'perm', None)}
            )
        else:
            x_std = (x - np.mean(x)) / np.std(x)
            N = len(x)
            rho = np.corrcoef(x, y)[0, 1]  # correlation coefficient between x and y
            best_eff = -np.inf  # efficiency is threshold_x / var_nht (the higher the better)
            best_eff_y = -np.inf
            for idx in trange(num_initial_nodes, desc="Generating initial designs"):
                if idx == 0:
                    perm = np.argsort(self.x)  # increasing order of x
                    #perm = self.rng.permutation(len(self.inclusions))

                elif idx % 2 == 0:
                    error = rng.normal(0, 1, N)
                    pseudo_y = rho * x_std + np.sqrt(1 - rho**2) * error
                    perm = np.argsort(pseudo_y)
                else:
                    perm = self.rng.permutation(len(self.inclusions))

                incl_perm = self.inclusions[perm]
                design = Design(
                    inclusions=incl_perm,
                    rng=self.rng,
                    perm=[int(v) for v in perm]
                )
                _ = self.criteria(design)
                var_nht = self.criteria.var_NHT
                var_nht_y = self.criteria.var_NHT_y
                eff = self.threshold_x / var_nht
                eff_y = float(self.threshold_y / var_nht_y)
                self.initial_designs.append(design)
                self.initial_nodes.append((var_nht, idx, design))
                self.initial_info.append(
                    {"var_nht": var_nht, "var_nht_y": var_nht_y, "perm": perm}
                )

                # Print the very first efficiency
                if idx == 0:
                    print(f"Initial design 0: efficiency x={eff:.4f}, y={eff_y:.4f}")

                # Print if this is the best so far
                if eff > best_eff:
                    print(f"New best at idx={idx}: efficiency x = {eff:.4f} and efficiency y = {eff_y:.4f}")
                    best_eff = eff
                # if eff_y > best_eff_y:
                #     print(f"New best at idx={idx}: efficiency x = {eff:.4f} and efficiency y = {eff_y:.4f}")
                #     best_eff_y = eff_y
                
            print()
            k = initial_design_to_use or 1
            sort_idx = np.argsort([node[0] for node in self.initial_nodes])[:k]
            self.initial_designs = [self.initial_designs[i] for i in sort_idx]
            self.initial_nodes = [self.initial_nodes[i] for i in sort_idx]
            self.initial_info = [self.initial_info[i] for i in sort_idx]

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
        new_design.merge_identical()
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
        random_restart_period: int = 200,
        random_injection_count: int = 2,
        prune_fraction: float = 0.9,
    ):
        initial_efficiency_x = np.round(self.threshold_x / self.initial_var_NHT, 3)
        initial_efficiency_y = np.round(self.threshold_y / self.initial_var_NHT_y, 3)

        closed_set = set()
        open_heap = []
        open_set = set()
        counter = itertools.count()

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

                tie_id = next(counter)
                heapq.heappush(open_heap, (new_criteria_value, tie_id, new_design))
                open_set.add(new_design)
                if len(open_heap) > max_open_set_size:
                    _, _, removed_design = heapq.heappop(open_heap)
                    open_set.discard(removed_design)

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

            # === RANDOM RESTART/RE-INJECTION ===
            if it > 0 and it % random_restart_period == 0:
                for _ in range(random_injection_count):
                    perm = self.rng.permutation(len(self.inclusions))
                    incl_perm = self.inclusions[perm]
                    perm_list = [int(v) for v in perm]
                    new_design = Design(
                        inclusions=incl_perm,
                        rng=self.rng,
                        perm=perm_list
                    )

                    _ = self.criteria(new_design)
                    var_nht = self.criteria.var_NHT
                    tie_id = next(counter)
                    heapq.heappush(open_heap, (var_nht, tie_id, new_design))
                    open_set.add(new_design)
                if show_results:
                    print(f"\nInjected {random_injection_count} random designs (restart) at iter {it}.")

            if len(open_heap) > max_open_set_size:
                n_keep = max(int(len(open_heap) * prune_fraction), 1)
                open_heap.sort()
                open_heap = open_heap[:n_keep]
                open_set = set(tup[2] for tup in open_heap)
                heapq.heapify(open_heap)
                if show_results:
                    print(f"\nPruned open heap; kept top {n_keep} nodes at iter {it}.")

            if self.best_criteria_value < (self.threshold_x * self.var_percent_exected):
                print("\nEarly stopping due to threshold!\n")
                if show_results == 1:# and (it % 10 == 0 or it == max_iterations - 1):
                    print(
                        f"\n=== Best Solution Updated at Iteration {it} ===\n"
                        f"  Best Cost (x):       {np.round(self.best_cost, 3)}\n"
                        f"  Best Cost (y):       {np.round(self.best_cost_y, 4)}\n"
                        f"  Criteria Value:      {np.round(self.best_criteria_value, 3)}\n"
                        f"  Efficiency x (0→f):  {initial_efficiency_x} → {np.round(self.threshold_x / self.best_cost, 3)}\n"
                        f"  Efficiency y (0→f):  {initial_efficiency_y} → {np.round(self.threshold_y / self.best_cost_y, 4)}\n"
                        f"  Alpha:               {self.switch_coefficient}\n"
                        f"  Design Depth:        {getattr(self.best_design, 'changes', 'NA')}\n"
                        f"  Design Size (|D|):   {len(getattr(self.best_design, 'heap', []))}\n"
                        f"  Open set size:       {len(open_set)}\n"
                    )
                return it
            if show_results == 1 and (it == 1 or it == max_iterations - 1):
                        print(
                            f"\n=== Best Solution Updated at Iteration {it} ===\n"
                            f"  Best Cost (x):       {np.round(self.best_cost, 3)}\n"
                            f"  Best Cost (y):       {np.round(self.best_cost_y, 4)}\n"
                            f"  Efficiency x (0→f):  {initial_efficiency_x} → {np.round(self.threshold_x / self.best_cost, 7)}\n"
                            f"  Efficiency y (0→f):  {initial_efficiency_y} → {np.round(self.threshold_y / self.best_cost_y, 7)}\n"
                            f"  Alpha:               {self.switch_coefficient}\n"
                            f"  Design Depth:        {getattr(new_design, 'changes', 'NA')}\n"
                            f"  Design Size (|D|):   {len(getattr(new_design, 'heap', []))}\n"
                            f"  Open set size:       {len(open_set)}\n"
                        )

        return max_iterations