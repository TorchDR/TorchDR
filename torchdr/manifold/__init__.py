"""
All manifolds and optimizers for Riemannian optimization.
"""
# Author:   Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License


from .manifold import (
    Manifold,
    ManifoldParameter,
)

from .poincare import (
    PoincareBall,
)

from .euclidean import (
    Euclidean,
)

from .radam import (
    RiemannianAdam
)

__all__ = [
    "Manifold",
    "ManifoldParameter",
    "Euclidean",
    "PoincareBall",
    "RiemannianAdam",
]
