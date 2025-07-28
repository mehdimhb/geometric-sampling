from importlib import metadata
import warnings

from .design import Design
from .new_design import NewDesign
from .population import Population
from . import criteria
from . import search
from . import sampling
from . import clustering
from . import random
from . import measure


__version__ = metadata.version("graphical_sampling")

__all__ = [
    "Population",
    "Design",
    "NewDesign",
    "criteria",
    "search",
    "sampling",
    "clustering",
    "random",
    "measure",
]


warnings.filterwarnings(
    "ignore",
    message=r'Environment variable "R_SESSION_TMPDIR" redefined by R.*',
    category=UserWarning,
    module='rpy2.rinterface.__init__'
)

