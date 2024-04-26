import numpy as np
from kernels import Kernel


def kernel_smoothing(Kernel:Kernel, x_grid:np.ndarray, u_data:np.ndarray,
                     f_data:np.ndarray, alpha_list:list[np.ndarray], nugget=1e-10) -> tuple[np.ndarray, np.ndarray]:
    """    Perform kernel smoothing on the data and estimate multi-derivatives.



    Return the smoothed data and the estimated derivatives. Each grid point
    will contribute one vector. u_smoothed is n_data x n_grid, and
    multi_derivatives is n_data x n_grid x n_alpha.

    Args:
        kernel (Kernel): Kernel function to use
        x_grid (np.ndarray): Grid points
        u_data (np.ndarray): Training data. Values of the solution at the grid points.
        f_data (np.ndarray): Training data. Output values f(x).
        alpha_list[np.ndarray]: List of alpha values for the multi-derivatives to calculate
        nugget (float): Regularization parameter. Default is 1e-10.

    Returns:
        u_smoothed (np.ndarray): Smoothed data
        multi_derivatives (np.ndarray): Estimated multi-derivatives
    """
    n_grid = len(x_grid)
    n_data = len(u_data)
    n_alpha = len(alpha_list)

    # Calculate the kernel matrix K using the provided kernel object
    K = np.zeros((n_grid, n_data))
    for i in range(n_grid):
        for j in range(n_data):
            K[i, j] = kernel(x_grid[i], u_data[j])

    # Regularize and invert the kernel matrix
    K_reg = K + nugget * np.eye(n_grid)

    # Compute smoothed values u_smoothed using the regularized kernel matrix
    u_smoothed = np.dot(K_reg, u_data)

    # Initialize the array for multi-derivatives
    multi_derivatives = np.zeros((n_data, n_grid, n_alpha))

    # Compute multi-derivatives for each alpha in alpha_list
    for idx, alpha in enumerate(alpha_list):
        for i in range(n_grid):
            for j in range(n_data):
                multi_derivatives[j, i, idx] = kernel.multiDerivative(x_grid[i], u_data[j], alpha)

    return u_smoothed, multi_derivatives

    raise NotImplementedError("kernel_smoothing not implemented")


def assemble_features(u_smoothed, multi_derivatives, function_list)-> np.ndarray:
    """    Assemble the features for the optimization problem.

    Provide a list of lambda functions to generate the features.
    Functions should have the call signature f(u, multi_derivatives), and return
    the feature that is to be contributed to s.

    Features must be linearly independent and not identically zero.

    Returns a 3d tensor of features, of shape n_data x n_grid x n_features.

    Args:
        u_smoothed (np.ndarray): smoothed solution data from kernel smoothing.
        multi_derivatives (ndarray): multi derivative data from kernel smoothing.
        function_list (list[function]): List of functions to generate features.
    """

    raise NotImplementedError("assemble_features not implemented")


def learn_DE_form(kernel:Kernel, s_features:np.ndarray, f_labels:np.ndarray, nugget:float) -> function:
    """    Learn the operator from the features and labels.

    Use the kernel method to learn a representation of the operator. Return a function
    that can be used to evaluate the operator at new points.

    Args:
        kernel (Kernel): _description_
        s_features (np.ndarray): _description_
        f_labels (np.ndarray): _description_
        nugget (float): _description_
    """

    raise NotImplementedError("learn_operator not implemented")


def kernelized_DE_fit(kernel:Kernel, DE_operator:function, u_data:np.ndarray,
                      f_data:np.ndarray, nugget_f:float, nugget_boundary:float) -> np.ndarray:
    """    Fit a kernelized operator to the data.

    Use the kernel method to fit an operator to the data. Return a function
    that can be used to evaluate the operator at new points.

    Implement and solve equation 17 from the paper.

    Args:
        kernel (Kernel): _description_
        DE_operator (function): _description_
        u_data (np.ndarray): _description_
        f_data (np.ndarray): _description_
        nugget_f (float): _description_
        nugget_boundary (float): _description_

    Returns:
        np.ndarray: Solution weights for the kernelized operator.
    """

    raise NotImplementedError("kernelized_DE_fit not implemented")
