from dataclasses import dataclass
from typing import Generator, Any
import bisect

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
        # Using a list to simulate the open set, kept sorted by criteria_value.
        # This allows efficient access to min (open_set_list[0]) and max (open_set_list[-1]).
        open_set_list: list[Node] = []
        for i in range(len(self.initial_designs)):
            bisect.insort_left(open_set_list, Node(float(self.initial_criteria_value[i]), self.initial_designs[i]))

        for it in range(max_iterations):
            if not open_set_list:
                # No more designs to explore
                break

            # Get the node with the minimum criteria value (first element in the sorted list)
            # This operation is O(N) as it requires shifting elements.
            current_node = open_set_list.pop(0)
            current_design = current_node.design

            print('Criteria of current node:', current_node.criteria_value)

            # The commented-out closed_set logic suggests designs might be re-evaluated.
            # If a design should only be processed once to avoid redundant computations
            # or infinite loops in certain graph structures, uncomment and manage a closed_set.
            # However, for a simple replacement and to match the provided structure,
            # we keep it commented out as it was in the original snippet.
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
                new_node = Node(new_criteria_value, new_design)

                # Manage the open_set_list size based on max_open_set_size
                if len(open_set_list) < max_open_set_size:
                    # If the list is not full, simply insert the new node in its sorted position.
                    bisect.insort_left(open_set_list, new_node)
                else:
                    # If the list is full, check if the new node is "better" (has a smaller criteria_value)
                    # than the current "worst" node in the open set (which is the last element).
                    # 'mx' here represents the node with the largest criteria_value in the open set.
                    mx = open_set_list[-1] if open_set_list else None

                    # If the new node is better than the current worst in the set,
                    # remove the worst and add the new one.
                    # The original RedBlackTree had 'mx.criteria_value > new_criteria_value'
                    # implying we want to keep smaller criteria values.
                    if mx is None or new_node.criteria_value < mx.criteria_value:
                        if mx is not None:
                            # Remove the last element (the one with the largest criteria_value)
                            open_set_list.pop()
                            # Insert the new node in its correct sorted position.
                        bisect.insort_left(open_set_list, new_node)

                # Update the overall best design found so far
                if new_criteria_value < self.best_criteria_value:
                    self.best_design = new_design
                    self.best_criteria_value = new_criteria_value

                    print('\n==============================================')
                    print(f'New best criteria value: {self.best_criteria_value}')

                    # If the best criteria value is below the threshold, we can stop.
                    if self.best_criteria_value < self.threshold:
                        return it  # Return the current iteration count

        # If max_iterations are reached or open_set_list becomes empty
        return max_iterations

