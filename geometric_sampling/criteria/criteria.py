from dataclasses import dataclass
from abc import abstractmethod

from numpy.typing import NDArray

from ..design import Design


@dataclass
class Criteria:
    auxiliary_variable: NDArray
    inclusions: NDArray

    @abstractmethod
    def __call__(self, design: Design) -> float: ...