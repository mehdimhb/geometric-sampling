from __future__ import annotations

from typing import Iterator, Collection, Optional

import numpy as np
from matplotlib import pyplot as plt
from .structs import MaxHeap, Range


class Design:
    def __init__(
        self,
        inclusions: Optional[Collection[float]] = None,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.heap = MaxHeap[Range](rng=rng)
        self.rng = rng
        self.changes = 0
        if inclusions is not None:
            self.push_initial_design(inclusions)

    def push_initial_design(self, inclusions: Collection[float]):
        events: list[tuple[float, str, int]] = []
        level: float = 0
        for i, p in enumerate(inclusions):
            next_level = level + p
            if next_level < 1 - 1e-9:
                events.append((level, "start", i))
                events.append((next_level, "end", i))
                level = next_level
            elif next_level > 1 + 1e-9:
                events.append((level, "start", i))
                events.append((1, "end", i))
                events.append((0, "start", i))
                events.append((next_level - 1, "end", i))
                level = next_level - 1
            else:
                events.append((level, "start", i))
                events.append((1, "end", i))
                level = 0

        events.sort()
        active = set()
        last_point: float = 0

        for point, event_type, bar_index in events:
            if event_type == "start":
                active.add(bar_index)
            elif event_type == "end":
                if last_point != point:
                    self.push(Range(round(point - last_point, 9), frozenset(active)))
                active.remove(bar_index)

            last_point = point

    def copy(self) -> Design:
        new_design = Design(
            rng=self.rng,
        )
        new_design.heap = self.heap.copy()
        new_design.changes = self.changes
        return new_design

    def pull(self, random: bool = False) -> Range:
        if random:
            return self.heap.randompop()
        return self.heap.pop()

    def push(self, *args: Range) -> None:
        for r in args:
            if not r.almost_zero():
                self.heap.push(r)

    def merge_identical(self):
        dic = {}
        for r in self.heap:
            dic.setdefault(r.ids, 0)
            dic[r.ids] += r.length
        self.heap = MaxHeap[Range](
            initial_heap=[Range(length, ids) for ids, length in dic.items()],
            rng=self.rng,
        )

    def switch(
        self,
        r1: Range,
        r2: Range,
        coefficient: float = 0.5,
    ) -> tuple[Range, Range, Range, Range]:
        length = coefficient * min(r1.length, r2.length)
        n1 = self.rng.choice(list(r1.ids - r2.ids))
        n2 = self.rng.choice(list(r2.ids - r1.ids))
        return (
            Range(length, r1.ids - {n1} | {n2}),
            Range(r1.length - length, r1.ids),
            Range(length, r2.ids - {n2} | {n1}),
            Range(r2.length - length, r2.ids),
        )

    def iterate(self, random_pull: bool = False, switch_coefficient: float = 0.5) -> None:
        r1 = self.pull(random_pull)
        r2 = self.pull(random_pull)
        if r1.ids == r2.ids:
            self.push(Range(r1.length + r2.length, r1.ids))
        else:
            self.push(*self.switch(r1, r2, switch_coefficient))
        self.changes += 1

    def show(self) -> None:
        initial_level: float = 0
        for r in self.heap:
            for i in r.ids:
                plt.plot([i, i], [initial_level, initial_level + r.length])
            initial_level += r.length
        plt.show()

    def __iter__(self) -> Iterator[Range]:
        return iter(self.heap)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Design):
            return NotImplemented
        return self.heap == other.heap

    def __hash__(self) -> int:
        return hash(self.heap)
