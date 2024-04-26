import pytest
import numpy as np

from KBio.dataSimulators import rectangular_grid


def test_rectangular_grid_small():
    grid_small = rectangular_grid([-1, -1], [1,1], [3,3])
    assert grid_small.grid_tensors[0].shape == (3,3)
    assert grid_small.grid_tensors[1].shape == (3,3)

    # Check that the grid is correct
    # The first grid should be:
    x = np.array([-1, 0, 1])
    y = np.array([-1, 0, 1])
    X, Y = np.meshgrid(x, y)
    assert np.allclose(grid_small.grid_tensors[0], X)
    assert np.allclose(grid_small.grid_tensors[1], Y)
