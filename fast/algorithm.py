from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.typing import NDArray
import heapq
import portion as P
from matplotlib import pyplot as plt
from tqdm import tqdm

rng = np.random.default_rng()


class Design:

    def __init__(self, inclusions: list[float] = None) -> None:
        self.heap: list[tuple[float, set[int]]] = []
        if inclusions is not None:
            self.push_initial_design(inclusions)

    def push_initial_design(self, inclusions: list[float]):
        bars = []
        level = 0
        for p in inclusions:
            next_level = level + p
            if next_level < 1-1e-9:
                interval = P.closed(level, next_level)
                level = next_level
            elif next_level > 1+1e-9:
                interval = P.closed(level, 1) | P.closed(0, next_level-1)
                level = next_level-1
            else:
                interval = P.closed(level, 1)
                level = 0
            bars.append(interval)

        events = []
        for i, bar in enumerate(bars):
            for interval in bar:
                events.append((interval.lower, 'start', i))
                events.append((interval.upper, 'end', i))

        events.sort()

        active = set()
        last_point = 0

        for point, event_type, bar_index in events:
            if event_type == 'start':
                active.add(bar_index)
            elif event_type == 'end':
                if last_point != point:
                    self.push((round(point - last_point, 9), set(active)))
                active.remove(bar_index)

            last_point = point

    def copy(self) -> Design:
        new_design = Design()
        new_design.heap = self.heap[:]
        return new_design

    def pull(self) -> tuple[float, set[int]]:
        item = heapq.heappop(self.heap)
        return -item[0], item[1]

    def push(self, *args: tuple[float, set[int]]) -> None:
        for arg in args:
            if arg[0] < 1e-9:
                continue
            inverse_arg = (-arg[0], arg[1])
            heapq.heappush(self.heap, inverse_arg)

    def merge_identical(self):
        heap_copy = self.heap[:]
        self.heap.clear()
        dic = {}
        for prob, IDs in tqdm(heap_copy):
            prob = -prob
            h = tuple(sorted(list(IDs)))
            dic.setdefault(h, 0)
            dic[h] += prob
        for IDs, prob in dic.items():
            self.push((prob, set(IDs)))

    def show(self) -> None:
        heap_copy = self.heap[:]
        initial_level = 0
        while heap_copy:
            prob, IDs = heapq.heappop(heap_copy)
            prob = -prob
            for i in IDs:
                plt.plot([i, i], [initial_level, initial_level+prob])
            initial_level += prob
        plt.show()

def generate_design(design: Design, num_changes: int, length_function: Callable[[float, float], float]) -> Design:
    new_design = design.copy()
    for _ in tqdm(range(num_changes)):
        sample_1_prob, sample_1_IDs = new_design.pull()
        sample_2_prob, sample_2_IDs = new_design.pull()
        if sample_1_IDs == sample_2_IDs:
            new_design.push((sample_1_prob+sample_2_prob, sample_1_IDs))
        else:
            length = length_function(sample_1_prob, sample_2_prob)
            n1 = rng.choice(list(sample_1_IDs - sample_2_IDs))
            n2 = rng.choice(list(sample_2_IDs - sample_1_IDs))
            new_design.push(
                (length, sample_1_IDs-{n1}|{n2}),
                (sample_1_prob-length, sample_1_IDs),
                (length, sample_2_IDs-{n2}|{n1}),
                (sample_2_prob-length, sample_2_IDs)
            )
    return new_design

# TODO
# make bars out of design
# make sample from design
x = [0.07, 0.04, 0.03, 0.06] * 5_000 + [0.02] * 5_000
print(sum(x), len(x))
d = Design(x)
# d.show()
dd = generate_design(d, 100, lambda x, y: min(x, y)/2)
print(len(dd.heap))
dd.merge_identical()
print(len(dd.heap))
# dd.show()