from dataclasses import dataclass
from typing import Optional, Collection, Any

import numpy as np
from numpy._typing import NDArray
from tqdm import tqdm

from fast.algorithm import Design
from fast.red_black_tree import RedBlackTree
from fast.type import Comparable


@dataclass
class Criteria:
    var_NHT: float
    var_NHT_y: float
    var_NHT_yr: float
    NHT_estimator: NDArray
    NHT_estimator_y: NDArray
    NHT_yr_Bias: float
    NHT_y_Bias: float


@dataclass
class Node(Comparable):
    design: Design
    criteria: Criteria

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.criteria.var_NHT < other.criteria.var_NHT

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return (
            self.criteria.var_NHT == other.criteria.var_NHT
            and self.design == other.design
        )


class AStarFast:
    def __init__(
        self,
        x: NDArray,
        y: NDArray,
        inclusions: NDArray,
        threshold_x: float = 1e-2,
        threshold_y: float = 1e-2,
        length: float = 1e-5,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        self.x = x
        self.y = y
        self.inclusions = inclusions
        self.threshold_y = threshold_y
        self.threshold_x = threshold_x
        self.length = length
        self.rng = rng

        self.best_design: Optional[Design] = None
        self.best_criteria: Optional[Criteria] = None

    def criteria(self, design: Design) -> Criteria:
        NHT_estimator = np.array(
            [
                np.sum(self.x[list(sample.ids)] / self.inclusions[list(sample.ids)])
                for sample in design
            ]
        )
        NHT_estimator_y = np.array(
            [
                np.sum(self.y[list(sample.ids)] / self.inclusions[list(sample.ids)])
                for sample in design
            ]
        )
        probabilities = np.array([sample.length for sample in design])
        var_NHT = np.sum((NHT_estimator - np.sum(self.x)) ** 2 * probabilities)
        var_NHT_y = np.sum((NHT_estimator_y - np.sum(self.y)) ** 2 * probabilities)
        var_NHT_yr = np.sum(
            (NHT_estimator_y * np.sum(self.x) / NHT_estimator - np.sum(self.y)) ** 2
            * probabilities
        )
        NHT_yr_Bias = (
            np.sum(probabilities * NHT_estimator_y * np.sum(self.x) / NHT_estimator)
            - np.sum(self.y)
        ) / np.sum(self.y)
        NHT_y_Bias = (
            np.sum(NHT_estimator_y * probabilities) - np.sum(self.y)
        ) / np.sum(self.y)
        return Criteria(
            var_NHT,
            var_NHT_y,
            var_NHT_yr,
            NHT_estimator,
            NHT_estimator_y,
            NHT_yr_Bias,
            NHT_y_Bias,
        )

    @staticmethod
    def iterate_design(design: Design, num_changes: int) -> Design:
        new_design = design.copy()
        for _ in range(num_changes):
            new_design.iterate()
        return new_design

    def neighbors(
        self, design: Design, num_new_nodes: int, num_changes: int
    ) -> Collection[Design]:
        return [self.iterate_design(design, num_changes) for _ in range(num_new_nodes)]

    def run(
        self,
        max_iterations: int,
        num_new_nodes: int,
        max_open_set_size: int,
        num_changes: int,
    ):
        closed_set = set()
        open_set = RedBlackTree[Node]()

        self.best_design = Design(self.inclusions, rng=self.rng)
        self.best_criteria = self.criteria(self.best_design)
        open_set.insert(Node(self.best_design, self.best_criteria))

        for it in tqdm(range(max_iterations)):
            if not open_set:
                break
            node = open_set.get_min()
            if not node:
                break
            current_design = node.design
            if current_design in closed_set:
                continue
            closed_set.add(current_design)
            for new_design in self.neighbors(
                current_design, num_new_nodes, num_changes
            ):
                new_criteria = self.criteria(new_design)
                if new_design in closed_set:
                    continue

                new_cost = new_criteria.var_NHT + self.rng.random() * 0.0000001
                if len(open_set) < max_open_set_size:
                    open_set.insert(Node(new_design, new_criteria))
                else:
                    mx = open_set.get_max()
                    if mx is None or mx.criteria.var_NHT > new_cost:
                        if mx is not None:
                            open_set.remove(mx)
                        open_set.insert(Node(new_design, new_criteria))

                if new_cost < self.best_criteria.var_NHT:
                    self.best_design = new_design
                    self.best_criteria = new_criteria

                    if self.best_criteria.var_NHT < self.threshold_x:
                        return it
        return max_iterations
