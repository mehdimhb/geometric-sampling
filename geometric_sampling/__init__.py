from importlib import metadata

from .design import Design
from . import criteria
from . import search
from . import shatranj
from . import balanced_kmeans


__version__ = metadata.version("geometric_sampling")

__all__ = ["Design", "criteria", "search", "shatranj", "balanced_kmeans"]
