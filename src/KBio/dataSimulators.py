from typing import Any, Union
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from scipy import stats as st

from .simulators.SIS import SIS

class Grid(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.y_values = None
        # Meshgrid output
        self.grid_tensors = None
        # self.grid_list[i] = x_i
        self.grid_list = []
        # self.value_list[i] = y_i
        self.value_list = []
        # self.value_store[x_index] = y
        self.value_store = {}

    @abstractmethod
    def _grid_to_points(self) -> None:
        pass

    @abstractmethod
    def _fill_value_store(self) -> None:
        """
        Iterate over the value_list and fill value_store with key-value pairs
        """

        pass

    @abstractmethod
    def grid_generator(self, quiet=True) -> Union[None, NDArray[np.float_]]:
        pass

    @abstractmethod
    def fill_yvalues(self, function) -> None:
        """ Fill y_values with simulated data
        """
        for point in self.grid_list:
            self.value_list.append(function(point))

class rectangular_grid(Grid):
    def __init__(self, mins:np.ndarray, maxes:list[float], n_pts:list[int]) -> None:
        super().__init__()
        assert len(mins) == len(maxes) == len(n_pts), "Length of mins, maxes, and n_pts must be equal"
        self.mins = mins
        self.maxes = maxes
        self.dims = len(mins)
        self.n_pts = n_pts
        self.grid_generator()


    def grid_generator(self, quiet=True) -> Union[None, NDArray[np.float_]]:
        self.grid_tensors = np.meshgrid(*[np.linspace(min, max, n) for min, max, n in zip(self.mins, self.maxes, self.n_pts)])
        if not quiet:
            return self.grid.copy()
        else:
            return None

    def _grid_to_points(self) -> None:
        """ Convert grid to list of points
        """
        assert self.grid is not None, "Grid is not defined yet. Please define it first by calling self.grid()."

        self.grid_list = []
        for i in range(np.product(self.n_pts)):
            # Select the i'th point in the grid
            self.grid_list.append(tuple(grid_tensor.flat[i] for grid_tensor in self.grid_tensors))

    def _fill_value_store(self) -> None:
        """ Fill value_store with key-value pairs
        """
        assert self.grid_list is not None, "Grid list is not defined yet. Please define it first by calling self._grid_to_points()."
        assert self.value_list is not None, "Value list is not defined yet. Please define it first by calling self._fill_value_list()."

        for point in self.grid_list:
            self.value_store[point] = None

    def fill_yvalues(self, function) -> None:
        """ Fill y_values with simulated data
        """
        for point in self.grid_list:
            fx = function(point)
            self.value_list.append(fx)
            self.value_store[point] = fx

class LatinHyperCube(Grid):
    def __init__(self, mins:np.ndarray, maxes:list[float], n_pts:int, seed=None) -> None:
        super().__init__()
        assert len(mins) == len(maxes), "Length of mins, maxes, and n_pts must be equal"
        self.mins = mins
        self.maxes = maxes
        self.dims = len(mins)
        self.n_pts = n_pts
        self.seed = seed
        self.grid_generator()

    def grid_generator(self, quiet=True) -> Union[None, NDArray[np.float_]]:

        # perform latin hypercube sampling on the grid. Sample n_pts points
        self.grid = st.qmc.LatinHyperCube(dim=self.dims).random(self.n_pts)
        # Now rescale each dimension to the desired range.
        for i in range(self.dims):
            self.grid[:, i] = self.mins[i] + self.grid[:, i] * (self.maxes[i] - self.mins[i])

        if not quiet:
            return self.grid.copy()
        else:
            return None

    def _grid_to_points(self) -> None:
        """ Convert grid to list of points
        """
        assert self.grid is not None, "Grid is not defined yet. Please define it first by calling self.grid()."

        self.grid_list = []
        for i in range(self.n_pts):
            self.grid_list.append(tuple(self.grid[i]))


class DataSimulator(ABC):
    """ Base class for all data simulator classes

    ```
    simulator = DataSimulator()
    y_grid = simulator(grid)
    ```

    Args:
        ABC (_type_): _description_
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, grid, *args, **kwargs):
        """ Simulate data.
        """
        pass

class SIS_sim(DataSimulator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, grid, *args: Any, **kwds: Any) -> Any:
        # Get a solution on grid.
        y = SIS(*args, **kwds)
        y = "" # Placeholder
        return y