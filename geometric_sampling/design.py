from __future__ import annotations

from typing import Iterator, Collection, Optional

import numpy as np
from matplotlib import pyplot as plt
from .structs import MaxHeap, Sample, SampleGenetic


class Design:
    def __init__(
        self,
        inclusions: Optional[Collection[float]] = None,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.heap = MaxHeap[Sample](rng=rng)
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
                    self.push(Sample(round(point - last_point, 9), frozenset(active)))
                active.remove(bar_index)

            last_point = point

    def copy(self) -> Design:
        new_design = Design(
            rng=self.rng,
        )
        new_design.heap = self.heap.copy()
        new_design.changes = self.changes
        return new_design

    def pull(self, random: bool = False) -> Sample:
        if random:
            return self.heap.randompop()
        return self.heap.pop()

    def push(self, *args: Sample) -> None:
        for r in args:
            if not r.almost_zero():
                self.heap.push(r)

    def merge_identical(self):
        dic = {}
        for r in self.heap:
            dic.setdefault(r.ids, 0)
            dic[r.ids] += r.probability
        self.heap = MaxHeap[Sample](
            initial_heap=[Sample(length, ids) for ids, length in dic.items()],
            rng=self.rng,
        )

    def switch(
        self,
        r1: Sample,
        r2: Sample,
        coefficient: float = 0.5,
    ) -> tuple[Sample, Sample, Sample, Sample]:
        length = coefficient * min(r1.probability, r2.probability)
        n1 = self.rng.choice(list(r1.ids - r2.ids))
        n2 = self.rng.choice(list(r2.ids - r1.ids))
        return (
            Sample(length, r1.ids - {n1} | {n2}),
            Sample(r1.probability - length, r1.ids),
            Sample(length, r2.ids - {n2} | {n1}),
            Sample(r2.probability - length, r2.ids),
        )

    def iterate(
        self, random_pull: bool = False, switch_coefficient: float = 0.5
    ) -> None:
        r1 = self.pull(random_pull)
        r2 = self.pull(random_pull)
        if r1.ids == r2.ids:
            self.push(Sample(r1.probability + r2.probability, r1.ids))
        else:
            self.push(*self.switch(r1, r2, switch_coefficient))
        self.changes += 1

    def show(self) -> None:
        initial_level: float = 0
        for r in self.heap:
            for i in r.ids:
                plt.plot([i, i], [initial_level, initial_level + r.probability])
            initial_level += r.probability
        plt.show()

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.heap)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Design):
            return NotImplemented
        return self.heap == other.heap

    def __hash__(self) -> int:
        return hash(self.heap)

class DesignGenetic:
    def __init__(
            self,
            inclusions: Optional[Collection[float]] = None,
            rng: np.random.Generator = np.random.default_rng(42),
    ):
        self.step = 0
        self.heap = MaxHeap[SampleGenetic](rng=rng)
        self.rng = rng
        self.changes = 0
        self.sucsses = 0
        # self.inclusions = inclusions
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
                events.append((round(next_level - 1, 9), "end", i))
                level = next_level - 1
            else:
                events.append((level, "start", i))
                events.append((1, "end", i))
                level = 0

        events.sort()
        active = set()
        last_point: float = 0
        counter:float = 0
        for point, event_type, bar_index in events:
            if event_type == "start":
                active.add(bar_index)
            elif event_type == "end":
                if last_point != point:
                    self.push(SampleGenetic(round(point - last_point, 9), frozenset(active), [counter,[]]))
                    counter += 1
                active.remove(bar_index)

            last_point = point

    def copy(self) -> DesignGenetic:
        new_design = DesignGenetic(
            rng=self.rng,
        )
        new_design.heap = self.heap.copy()
        new_design.changes = self.changes
        return new_design

    def pull(self, random: bool = False) -> SampleGenetic:
        if self.heap.is_empty():
            return SampleGenetic(0, frozenset(), [0, []])

        if random:
            return self.heap.randompop()
        return self.heap.pop()

    def push(self, *args: SampleGenetic) -> None:
        for r in args:
            if not r.almost_zero():
                self.heap.push(r)


    def switch(
            self,
            r1: SampleGenetic,
            r2: SampleGenetic,
            coefficient: float = 0.5,
            border_units: set[int] = None,
            partitions: dict[int, list[int]] = None,
            step: int = 0
    ) -> tuple[SampleGenetic, SampleGenetic, SampleGenetic, SampleGenetic]:

        border_units = border_units or set()


        eligible_r1 = r1.ids - r2.ids
        eligible_r2 = r2.ids - r1.ids

        eligible_r1 -= border_units
        eligible_r2 -= border_units

        if partitions:
            common_partitions = []
            for part_idx, part_ids in partitions.items():
                part_set = set(part_ids)
                part_eligible_r1 = eligible_r1 & part_set
                part_eligible_r2 = eligible_r2 & part_set
                if part_eligible_r1 and part_eligible_r2:
                    common_partitions.append((part_idx, part_eligible_r1, part_eligible_r2))

            if not common_partitions:
                return (
                    SampleGenetic(r1.probability, r1.ids, r1.index),
                    SampleGenetic(0, frozenset(), [0, []]),
                    SampleGenetic(r2.probability, r2.ids, r2.index),
                    SampleGenetic(0, frozenset(),[0, []]),
                )
            else:
                chosen_partition = self.rng.choice(common_partitions)
                _, eligible_r1, eligible_r2 = chosen_partition

        if not eligible_r1 or not eligible_r2:
            return (
                SampleGenetic(r1.probability, r1.ids, r1.index),
                SampleGenetic(0, frozenset(), [0, []]),
                SampleGenetic(r2.probability, r2.ids, r2.index),
                SampleGenetic(0, frozenset(), [0, []]),
            )

        length = coefficient * min(r1.probability, r2.probability)

        n1 = self.rng.choice(list(eligible_r1))
        n2 = self.rng.choice(list(eligible_r2))

        new_r1 = SampleGenetic(length, (r1.ids - {n1}) | {n2}, r1.index)
        remaining_r1 = SampleGenetic(r1.probability - length, r1.ids, [r1.index[0], r1.index[1] + [step]] )
        new_r2 = SampleGenetic(length, (r2.ids - {n2}) | {n1}, r2.index )
        remaining_r2 = SampleGenetic(r2.probability - length, r2.ids, [r2.index[0], r2.index[1] + [step]] )
        self.sucsses+=1
        return new_r1, remaining_r1, new_r2, remaining_r2

    def iterate(
            self,
            random_pull: bool = False,
            switch_coefficient: float = 0.5,
            partitions: dict[int, list[int]] = None,
            border_units: set[int] = None,

    ) -> None:

        r1 = self.pull(random_pull)
        r2 = self.pull(random_pull)

        if r1.ids == r2.ids:
            self.push(SampleGenetic(r1.probability + r2.probability, r1.ids, r1.index))
        else:
            new_samples= self.switch(
                r1,
                r2,
                coefficient=switch_coefficient,
                border_units=border_units,
                partitions=partitions,
                step=self.step
            )


            self.step+=1
            self.push(*new_samples)


        self.changes += 1

    def show(self) -> None:
        initial_level: float = 0
        for r in sorted(list(self.heap), key=lambda sampleGenetic: (sampleGenetic.index[0], tuple(sampleGenetic.index[1]), len(sampleGenetic.index[1]))):
            for i in r.ids:
                plt.plot([i, i], [initial_level, initial_level + r.probability])
            initial_level += r.probability
        plt.show()

    def __iter__(self) -> Iterator[SampleGenetic]:
        return iter(self.heap)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Design):
            return NotImplemented
        return self.index == other.index

    def __hash__(self) -> int:
        return hash(self.heap)
