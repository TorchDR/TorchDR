"""
Tests for visualization functions.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from torchdr.utils.visu import plot_poincare_disk, plot_disk


def test_plot_poincare_disk():
    """Test the plot_poincare_disk function."""
    fig, ax = plt.subplots()
    plot_poincare_disk(ax)
    assert len(ax.images) == 1, "An image should be added to the axes."
    plt.close(fig)


def test_plot_disk():
    """Test the plot_disk function."""
    fig, ax = plt.subplots()
    plot_disk(ax)

    # Check that a circle has been added
    assert any(isinstance(p, Circle) for p in ax.patches), (
        "A Circle patch should be added to the axes."
    )

    # Check axis limits
    assert ax.get_xlim() == (-1.05, 1.05)
    assert ax.get_ylim() == (-1.05, 1.05)

    plt.close(fig)
