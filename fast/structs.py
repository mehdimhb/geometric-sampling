from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Iterator


class MaxHeap[T]:
    def __init__(self):
        self.heap: list[T] = []

    def push(self, item: T):
        heapq.heappush(self.heap, -item)

    def pop(self) -> T:
        return -heapq.heappop(self.heap)

    def peek(self) -> T:
        return -self.heap[0]

    def copy(self) -> MaxHeap[T]:
        new_heap = MaxHeap()
        new_heap.heap = self.heap[:]
        return new_heap

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return bool(self.heap)

    def __iter__(self) -> Iterator[T]:
        return iter(self.heap)

    def __str__(self):
        return str(self.heap)

    def __hash__(self) -> int:
        return hash(tuple(self.heap))

    def __eq__(self, other: MaxHeap[T]) -> bool:
        return self.heap == other.heap


@dataclass
class Range:
    length: float
    ids: set[int]

    def almost_zero(self) -> bool:
        return self.length < 1e-9

    def __eq__(self, other: Range) -> bool:
        return self.ids == other.ids

    def __lt__(self, other: Range) -> bool:
        return self.length < other.length

    def __le__(self, other: Range) -> bool:
        return self.length <= other.length

    def __gt__(self, other: Range) -> bool:
        return self.length > other.length

    def __ge__(self, other: Range) -> bool:
        return self.length >= other.length

    def __neg__(self):
        return Range(-self.length, self.ids)

    def __add__(self, other: Range) -> Range:
        return Range(self.length + other.length, self.ids | other.ids)

    def __hash__(self):
        return hash(frozenset(self.ids))
