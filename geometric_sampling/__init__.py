from importlib import metadata

from .design import Design
from . import criteria
from . import search


__version__ = metadata.version("geometric_sampling")

__all__ = ["Design", "criteria", "search"]
