from abc import abstractmethod
from typing import Protocol, Any


class Comparable(Protocol):
    @abstractmethod
    def __eq__(self, other: Any) -> bool: ...

    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...


class Negatable(Protocol):
    def __neg__(self) -> "Negatable": ...


class ComparableNegatable(Comparable, Negatable, Protocol): ...
