from importlib import metadata

from .structs import Range, MaxHeap
from .design import Design
from .criteria.var_nht import VarNHT
from .criteria.criteria import Criteria
from .search_algorithms.astar import AStarFast

__version__ = metadata.version("geometric_sampling")

__all__ = [
    "Design",
    "Range",
    "MaxHeap",
    "VarNHT",
    "Criteria",
    "AStarFast",
]
