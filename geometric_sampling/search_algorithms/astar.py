from typing import Generator, Any

import numpy as np
from dataclasses import dataclass

from ..criteria.criteria import Criteria
from ..design import Design
from ..red_black_tree import RedBlackTree


@dataclass(frozen=True, order=False, eq=False)
class Node:
    criteria_value: float
    design: Design

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.criteria_value < other.criteria_value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.criteria_value == other.criteria_value

    def __le__(self, other: Any) -> bool:
        return self < other or self == other

    def __ge__(self, other: Any) -> bool:
        return not self < other

    def __gt__(self, other: Any) -> bool:
        return other < self


class AStarFast:
    def __init__(
        self,
        criteria: Criteria,
        threshold_x: float = 1e-2,
        threshold_y: float = 1e-2,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        self.threshold_y = threshold_y
        self.threshold_x = threshold_x
        self.rng = rng

        self.criteria = criteria
        self.best_design = Design(self.criteria.inclusions, rng=self.rng)
        self.best_criteria_value = self.criteria(self.best_design)

    @staticmethod
    def iterate_design(design: Design, num_changes: int) -> Design:
        new_design = design.copy()
        for _ in range(num_changes):
            new_design.iterate()
        return new_design

    def neighbors(
        self, design: Design, num_new_nodes: int, num_changes: int
    ) -> Generator[Design, None, None]:
        for _ in range(num_new_nodes):
            yield self.iterate_design(design, num_changes)

    def run(
        self,
        max_iterations: int,
        num_new_nodes: int,
        max_open_set_size: int,
        num_changes: int,
    ):
        closed_set = set()
        open_set = RedBlackTree[Node]()
        open_set.insert(Node(self.best_criteria_value, self.best_design))

        for it in range(max_iterations):
            if not open_set:
                break
            mn = open_set.get_min()
            if not mn:
                break
            current_design = mn.design
            if current_design in closed_set:
                continue
            closed_set.add(current_design)
            for new_design in self.neighbors(
                current_design, num_new_nodes, num_changes
            ):
                new_criteria_value = self.criteria(new_design)
                if new_design in closed_set:
                    continue
                if len(open_set) < max_open_set_size:
                    open_set.insert(Node(new_criteria_value, new_design))
                else:
                    mx = open_set.get_max()
                    if mx is None or mx.criteria_value > new_criteria_value:
                        if mx is not None:
                            open_set.remove(mx)
                        open_set.insert(Node(new_criteria_value, new_design))

                if new_criteria_value < self.best_criteria_value:
                    self.best_design = new_design
                    self.best_criteria_value = new_criteria_value

                    if self.best_criteria_value < self.threshold_x:
                        return it
        return max_iterations
