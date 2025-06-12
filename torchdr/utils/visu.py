# -*- coding: utf-8 -*-
"""
Useful functions to draw Hyperbolic outputs of TorchDr
"""

# Author: Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

import numpy as np
import matplotlib.pylab as plt


def plot_poincare_disk(ax, alpha=0.1):
    """
    Plot a Poincaré disk model with shading based on hyperbolic distance.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the Poincaré disk.
    alpha : float, optional
        The transparency level of the shading, by default 0.1.
    """
    # Create a grid of points
    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-1, 1, 1000)
    X, Y = np.meshgrid(x, y)

    # Calculate the distance from the origin
    distance = np.sqrt(X**2 + Y**2)
    hypDistance = np.arccosh(1 + 2 * (distance) / (1 - distance + 1e-10))
    # Define a shading function based on distance for the interior only
    radius = 1.0  # Radius of the disk
    interior = distance <= radius
    shading = np.zeros_like(distance)
    shading[interior] = hypDistance[interior]  # Shading based on distance
    ax.imshow(shading, extent=[-1, 1, -1, 1], cmap='Greys', alpha=alpha)


def plot_disk(ax, alpha=0.5):
    """
    Plot a grid on the Poincaré disk.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the grid.
    alpha : float, optional
        The transparency level of the shaded distance background, by default 0.5.
    """
    ax.set_aspect('equal')
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.axis('off')

    plot_poincare_disk(ax, alpha=alpha)
    circle = plt.Circle((0, 0), 1, color="k", linewidth=1, fill=False)
    ax.add_patch(circle)
