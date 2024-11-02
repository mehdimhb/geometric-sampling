from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Iterator, Generic, Collection, Optional, Any
from typing import TypeVar

import numpy as np

from .type import ComparableAndNeg

T = TypeVar("T", bound=ComparableAndNeg)


class MaxHeap(Generic[T]):
    def __init__(
        self,
        initial_heap: Optional[Collection[T]] = None,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.heap = []
        if initial_heap is not None:
            self.heap = [-item for item in initial_heap]
            heapq.heapify(self.heap)
        self.rng = rng

    def push(self, item: T):
        heapq.heappush(self.heap, -item)

    def pop(self) -> T:
        return -heapq.heappop(self.heap)

    def peek(self) -> T:
        return -self.heap[0]

    def randompop(self) -> T:
        idx = self.rng.integers(len(self.heap))
        val = -self.heap[idx]
        self.heap[idx] = self.heap[-1]
        self.heap.pop()
        if idx < len(self.heap):
            heapq._siftup(self.heap, idx)  # type: ignore
            heapq._siftdown(self.heap, 0, idx)  # type: ignore
        return val

    def copy(self) -> MaxHeap[T]:
        new_heap = MaxHeap[T]()
        new_heap.heap = self.heap[:]
        new_heap.rng = self.rng
        return new_heap

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return bool(self.heap)

    def __iter__(self) -> Iterator[T]:
        return map(lambda x: -x, self.heap)

    def __str__(self):
        return str(list(map(lambda x: -x, self.heap)))

    def __hash__(self) -> int:
        return hash(tuple(self.heap))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MaxHeap):
            return NotImplemented
        return self.heap == other.heap


@dataclass(order=False)
class Range(ComparableAndNeg):  # TODO: Remove ComparableAndNeg
    length: float
    ids: frozenset[int]

    def almost_zero(self) -> bool:
        return self.length < 1e-9

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return self.ids == other.ids

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return self.length < other.length

    def __neg__(self):
        return Range(-self.length, self.ids)

    def __add__(self, other: Range) -> Range:
        return Range(self.length + other.length, self.ids | other.ids)

    def __hash__(self):
        return hash(self.ids)
