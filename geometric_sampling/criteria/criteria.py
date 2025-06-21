from dataclasses import dataclass
from abc import abstractmethod

from numpy.typing import NDArray

from ..design import Design

@dataclass
class Criteria:
    main_variable: NDArray
    auxiliary_variable: NDArray
    inclusion_probability: NDArray
    balance_method: str

    @abstractmethod
    def __call__(self, design: Design) -> float: ...
