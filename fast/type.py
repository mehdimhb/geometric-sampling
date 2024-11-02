from abc import abstractmethod
from typing import TypeVar, Protocol, Any

C = TypeVar("C", bound="Comparable")


class Comparable(Protocol):
    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __lt__(self: C, other: Any) -> bool:
        pass

    def __gt__(self: C, other: Any) -> bool:
        return (not self < other) and self != other

    def __le__(self: C, other: Any) -> bool:
        return self < other or self == other

    def __ge__(self: C, other: Any) -> bool:
        return not self < other

    def __neg__(self):
        return NotImplemented
