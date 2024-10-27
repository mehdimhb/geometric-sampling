from __future__ import annotations

from typing import Iterator, Collection

import numpy as np
import portion as P
from matplotlib import pyplot as plt
from tqdm import tqdm
from fast.structs import MaxHeap, Range


class Design:
    def __init__(
        self,
        inclusions: Collection[float] = None,
        switch_coefficient: float = 0.5,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.heap = MaxHeap[Range]()
        self.switch_coefficient = switch_coefficient
        self.rng = rng
        self.changes = 0
        if inclusions is not None:
            self.push_initial_design(inclusions)

    # TODO: Refactor
    def push_initial_design(self, inclusions: Collection[float]):
        bars = []
        level = 0
        for p in inclusions:
            next_level = level + p
            if next_level < 1 - 1e-9:
                interval = P.closed(level, next_level)
                level = next_level
            elif next_level > 1 + 1e-9:
                interval = P.closed(level, 1) | P.closed(0, next_level - 1)
                level = next_level - 1
            else:
                interval = P.closed(level, 1)
                level = 0
            bars.append(interval)

        events = []
        for i, bar in enumerate(bars):
            for interval in bar:
                events.append((interval.lower, "start", i))
                events.append((interval.upper, "end", i))

        events.sort()

        active = set()
        last_point = 0

        for point, event_type, bar_index in events:
            if event_type == "start":
                active.add(bar_index)
            elif event_type == "end":
                if last_point != point:
                    self.push(Range(round(point - last_point, 9), frozenset(active)))
                active.remove(bar_index)

            last_point = point

    def copy(self) -> Design:
        new_design = Design()
        new_design.heap = self.heap.copy()
        new_design.rng = self.rng
        new_design.switch_coefficient = self.switch_coefficient
        new_design.changes = self.changes
        return new_design

    def pull(self) -> Range:
        return self.heap.pop()

    def push(self, *args: Range) -> None:
        for r in args:
            if not r.almost_zero():
                self.heap.push(r)

    def merge_identical(self):
        new_heap = MaxHeap[Range]()
        dic = {}
        for r in self.heap:
            dic.setdefault(r.ids, 0)
            dic[r.ids] += r.length
        for ids, length in dic.items():
            new_heap.push(Range(length, ids))
        self.heap = new_heap

    def switch(
        self,
        r1: Range,
        r2: Range,
    ) -> tuple[Range, Range, Range, Range]:
        length = self.switch_coefficient * min(r1.length, r2.length)
        n1 = self.rng.choice(list(r1.ids - r2.ids))
        n2 = self.rng.choice(list(r2.ids - r1.ids))
        return (
            Range(length, r1.ids - {n1} | {n2}),
            Range(r1.length - length, r1.ids),
            Range(length, r2.ids - {n2} | {n1}),
            Range(r2.length - length, r2.ids),
        )

    def iterate(self) -> None:
        r1 = self.pull()
        r2 = self.pull()
        if r1 == r2:
            self.push(r1 + r2)
        else:
            self.push(*self.switch(r1, r2))
        self.changes += 1

    def show(self) -> None:
        initial_level = 0
        for r in self.heap:
            for i in r.ids:
                plt.plot([i, i], [initial_level, initial_level + r.length])
            initial_level += r.length
        plt.show()

    def __iter__(self) -> Iterator[Range]:
        return iter(self.heap)

    def __eq__(self, other: Design) -> bool:
        return self.heap == other.heap

    def __hash__(self) -> int:
        return hash(self.heap)


def generate_design(design: Design, n: int) -> Design:
    new_design = design.copy()
    for _ in tqdm(range(n)):
        new_design.iterate()
    return new_design
