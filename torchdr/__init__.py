# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from .__about__ import (
    __title__,
    __summary__,
    __url__,
    __version__,
    __author__,
    __license__,
)

from . import spectral
from . import affinity_matcher
from . import affinity
from . import losses
from . import utils
from . import neighbor_embedding

__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__license__",
    "spectral",
    "affinity_matcher",
    "affinity",
    "losses",
    "utils",
    "neighbor_embedding",
]
