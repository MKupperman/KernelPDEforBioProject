from typing import Any, Union, Callable
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from scipy import stats as st
from scipy.interpolate import RegularGridInterpolator

from .simulators.SIS import SIS
from .simulators.Advection1D_Solver import solve_1D_advection

class Grid(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.y_values = None
        # Meshgrid output
        self.grid_tensors = None
        self.grid_tensors_values = None
        # self.grid_list[i] = x_i
        self.grid_list = []
        # self.value_list[i] = y_i
        self.value_list = []
        # self.value_store[x_index] = y
        self.value_store = {}

        self.grid_tensors_forcing = None
        # self.forcing_list[i] = f_i
        self.forcing_list = []

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

    def fill_yvalues(self, function) -> None:
        """ Fill y_values with simulated data
        """
        for point in self.grid_list:
            self.value_list.append(function(point))

    def fill_forcing(self, function) -> None:
        """ Fill forcing_list with forcing data
        """

        for point in self.grid_list:
            self.forcing_list.append(function(point))

class rectangular_grid(Grid):
    def __init__(self, mins:np.ndarray, maxes:list[float], n_pts:list[int]) -> None:
        super().__init__()
        assert len(mins) == len(maxes) == len(n_pts), "Length of mins, maxes, and n_pts must be equal"
        self.mins = mins
        self.maxes = maxes
        self.dims = len(mins)
        self.n_pts = n_pts
        self.grid_generator()
        self._grid_to_points()


    def grid_generator(self, quiet=True) -> Union[None, NDArray[np.float_]]:
        self.grid_tensors = np.meshgrid(*[np.linspace(min, max, n) for min, max, n in zip(self.mins, self.maxes, self.n_pts)])

        # Rework this for higher dimensions - flatten the grid tensors
        flat_grids = [grid.flatten() for grid in self.grid_tensors]
        num_pts = np.prod(self.n_pts)
        # Iterate over this many points

        for i in range(num_pts):
            self.grid_list.append(np.asarray([flat_grid[i] for flat_grid in flat_grids]))


        if not quiet:
            return self.grid.copy()
        else:
            return None

    def _grid_to_points(self) -> None:
        """ Convert grid to list of points
        """
        assert self.grid_list is not None, "Grid is not defined yet. Please define it first by calling self.grid()."

        self.grid_list = []
        for i in range(np.prod(self.n_pts)):
            # Select the i'th point in the grid
            t = tuple(grid_tensor.flat[i] for grid_tensor in self.grid_tensors)
            self.grid_list.append(t)

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
        self.grid_tensors_values = np.asarray(self.value_list).reshape(self.n_pts, order='F')



    def fill_forcing(self, function) -> None:
        super().fill_forcing(function)
        # Now fill the forcing values into the value store
        # for i, point in enumerate(self.grid_list):
        #     self.value_store[point] = self.forcing_list[i]

        # Check that this inflates the array correctly.
        self.grid_tensors_forcing = np.asarray(self.forcing_list).reshape(self.n_pts)

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
    def __init__(self, dt=1, S0=99, I0=1, beta=0.02,gamma=0.01,T_final=20,
                 forcing:callable = lambda x : 0) -> None:
        self.dt = dt
        self.S0 = S0
        self.I0 = I0
        self.beta = beta
        self.gamma = gamma
        self.T_final = T_final
        self.forcing = forcing
        super().__init__()

    def asymptotic_steady_state(self, beta=None, gamma=None):
        """ If No forcing is applied, where do we expect the system to settle?
        """
        if beta is None:
            beta = self.beta
        if gamma is None:
            gamma = self.gamma

        return np.maximum(0, 1 - gamma / beta)

    def __call__(self, grid:Grid, forcing:callable = None, verbose=False, *args: Any, **kwds: Any) -> Any:
            """Simulate data using the SIS model.

            Args:
                grid (Grid): The grid object representing the spatial domain.
                forcing (callable): The forcing function that determines the dynamics of the system.
                *args: Variable length argument list.
                **kwds: Arbitrary keyword arguments.

            Returns:
                Any: The simulated data.

            Raises:
                AssertionError: If the grid is not an instance of Grid.
                AssertionError: If T_final is not an integer or float.

            Notes:
                This function simulates data using the SIS (Susceptible-Infected-Susceptible) model. The model is a
                compartmental model used to study the spread of infectious diseases. The simulation is performed on a
                spatial grid defined by the `grid` object. The dynamics of the system are determined by the `forcing`
                function. Additional parameters can be provided as keyword arguments to override the default values set
                during instantiation.

            """
            # Do a bunch of checks on the input first

            # Check if grid is a Grid object
            assert isinstance(grid, Grid), "grid must be an instance of Grid"
            # Check if we got any kwargs to pass to SIS

            if "dt" not in kwds:
                dt = self.dt
            else:
                dt = kwds["dt"]

            if "S0" not in kwds:
                S0 = self.S0
            else:
                S0 = kwds["S0"]

            if "I0" not in kwds:
                I0 = self.I0
            else:
                I0 = kwds["I0"]

            if "beta" not in kwds:
                beta = self.beta
            else:
                beta = kwds["beta"]

            if "gamma" not in kwds:
                gamma = self.gamma
            else:
                gamma = kwds["gamma"]

            # Check that forcing is a callable
            if forcing is not None:
                assert callable(forcing), "forcing must be a callable object. A __call__ method must be available."

            if forcing is None:
                forcing = self.forcing

            if "T_final" not in kwds:
                T_final = self.T_final
                if T_final <= 0:
                    raise ValueError("T_final must be greater than 0")
                elif T_final < np.max(grid.grid_tensors):
                    raise ValueError("T_final must be greater than the maximum value of the grid. Reduce grid length or increase T_final")
            else:
                assert isinstance(T_final, (int, float)), "T_final must be an integer or float"
                T_final = kwds["T_final"]
            if verbose:
                print("Simulating data using the SIS model...")
            # Get a solution on grid.
            t_nodes, y_nodes = SIS(dt, S0, I0, beta, gamma, T_final, f=forcing)
            if verbose:
                print(f"Simulated data on grid of shape {t_nodes.shape}")
            # Now interpolate the solution down onto the grid.
            # scipy interp 1d here
            yfun = lambda x: np.interp(x, t_nodes, y_nodes)
            ffun = lambda x: np.interp(x, t_nodes, np.asarray([forcing(t) for t in t_nodes]))
            grid.fill_forcing(ffun)
            grid.fill_yvalues(yfun)


class Advection1D_sim(DataSimulator):
    def __init__(self, dt=1e-2, u0=lambda x: 1, ux = lambda x: 1,
                 T_final=1.0, forcing: callable = None,
                 x_min=0, x_max=1, nx=100) -> None:
        self.dt = dt
        self.u0 = u0  # initial condition of the solution
        self.ux = ux  # velocity field over the domain.
        self.T_final = T_final
        self.forcing = forcing  # source/sink forcing terms.
        self.nx = nx
        self.x_min = x_min
        self.x_max = x_max
        super().__init__()

    def __call__(self, grid: Grid, forcing: callable, verbose=False, *args: Any, **kwds: Any) -> Any:
        assert isinstance(grid, Grid), "grid must be an instance of Grid"
        if "dt" not in kwds:
            dt = self.dt
        else:
            dt = kwds["dt"]
        if "u0" not in kwds:
            u0 = self.u0
        else:
            print("Using custom u0")
            u0 = kwds["u0"]
        if "ux" not in kwds:
            ux = self.ux
        else:
            ux = kwds["ux"]
        if forcing is None:
            forcing = self.forcing
        if forcing is not None:
            assert callable(forcing), "forcing must be a callable object. A __call__ method must be available."
        if "T_final" not in kwds:
            T_final = self.T_final
        else:
            assert isinstance(T_final, (int, float)), "T_final must be an integer or float"
            T_final = kwds["T_final"]

        # check if we got a pyvis flag
        if "pyvis" not in kwds:
            pyvis = False
        else:
            pyvis = kwds["pyvis"]
            assert isinstance(pyvis, bool), "pyvis must be a boolean"
        if verbose:
            print("Simulating data using the 1D advection model...")
        x_grid = np.linspace(self.x_min, self.x_max, self.nx)
        t_nodes, u_nodes = solve_1D_advection(x_min=self.x_min, x_max=self.x_max, dt=dt,
                                              u0=u0, ux=ux, T_final=T_final, forcing=forcing,
                                              nx=self.nx, pyvis=pyvis)
        print(t_nodes.shape)
        print(u_nodes.shape)
        if verbose:
            print(f"Simulated data on grid of shape {t_nodes.shape}")

        def ffun(xpt):
            t, x = xpt
            if forcing is not None:
                return forcing(t, x)
            else:
                return 0
            # gridT, gridX = np.meshgrid(t_nodes, x_grid)
            # Compute the forcing at the point xpt

        interp_u = RegularGridInterpolator((t_nodes, x_grid), u_nodes)
        def ufun(xpt):
            t, x = xpt
            return interp_u((t, x))

        grid.fill_forcing(ffun)
        grid.fill_yvalues(ufun)

        return t_nodes, x_grid, u_nodes, ffun, ufun
