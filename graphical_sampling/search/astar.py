from dataclasses import dataclass
from typing import Generator, Any

import numpy as np

from ..criteria.criteria import Criteria
from ..new_design import NewDesign
from ..red_black_tree import RedBlackTree


@dataclass(frozen=True, order=False, eq=False)
class Node:
    criteria_value: float
    design: NewDesign

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


class AStar:
    def __init__(
        self,
        initial_designs: list[NewDesign],
        criteria: Criteria,
        *,
        threshold: float = -1.0,
    ) -> None:
        self.initial_designs = initial_designs
        self.criteria = criteria
        self.threshold = threshold
        self.rng = np.random.default_rng()

        self.initial_criteria_value = np.array([self.criteria(design) for design in self.initial_designs])
        self.best_design = self.initial_designs[np.argmax(self.initial_criteria_value)]
        self.best_criteria_value = np.min(self.initial_criteria_value)

        print('best initial criteria value', self.best_criteria_value)

    def iterate_design(
            self,
            design: NewDesign,
            n_clusters_to_change_order_zone: int | str = 'all',
            n_clusters_to_change_order_units: int | str = 'all',
            n_zones_to_change_order_units: int | str = 'all',
            n_changes_in_order_of_units: int = 1,
            n_changes_in_order_of_zones: int = 1
    ) -> NewDesign:
        new_design = design.copy()
        new_design.iterate(
            n_clusters_to_change_order_zone,
            n_clusters_to_change_order_units,
            n_zones_to_change_order_units,
            n_changes_in_order_of_units,
            n_changes_in_order_of_zones,
        )
        return new_design

    def neighbors(
        self,
        design: NewDesign,
        num_new_nodes: int,
        n_clusters_to_change_order_zone: int | str = 'all',
        n_clusters_to_change_order_units: int | str = 'all',
        n_zones_to_change_order_units: int | str = 'all',
        n_changes_in_order_of_units: int = 1,
        n_changes_in_order_of_zones: int = 1,
    ) -> Generator[NewDesign, None, None]:
        for _ in range(num_new_nodes):
            yield self.iterate_design(
                design,
                n_clusters_to_change_order_zone,
                n_clusters_to_change_order_units,
                n_zones_to_change_order_units,
                n_changes_in_order_of_units,
                n_changes_in_order_of_zones,
            )

    def run(
        self,
        max_iterations: int,
        num_new_nodes: int,
        max_open_set_size: int,
        n_clusters_to_change_order_zone: int | str = 'all',
        n_clusters_to_change_order_units: int | str = 'all',
        n_zones_to_change_order_units: int | str = 'all',
        n_changes_in_order_of_units: int = 1,
        n_changes_in_order_of_zones: int = 1,
    ):
        closed_set = set()
        open_set = RedBlackTree[Node]()
        for i in range(len(self.initial_designs)):
            open_set.insert(Node(float(self.initial_criteria_value[i]), self.initial_designs[i]))

        for it in range(max_iterations):
            if not open_set:
                break
            mn = open_set.get_min()
            if not mn:
                break
            current_design = mn.design
            # if current_design in closed_set:
            #     continue
            # closed_set.add(current_design)
            for new_design in self.neighbors(
                current_design,
                num_new_nodes,
                n_clusters_to_change_order_zone,
                n_clusters_to_change_order_units,
                n_zones_to_change_order_units,
                n_changes_in_order_of_units,
                n_changes_in_order_of_zones,
            ):
                new_criteria_value = self.criteria(new_design)

                print('new criteria value:', new_criteria_value)

                # if new_design in closed_set:
                #     continue
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

                    print('\n==============================================')
                    print(f'New best criteria value: {self.best_criteria_value}')

                    if self.best_criteria_value < self.threshold:
                        return it
        return max_iterations
