from typing import Callable

import numpy as np
from .kernels import Kernel
from .optim import regularized_cholesky_solve

def kernel_smoothing(kernel:Kernel, x_grid:np.ndarray, u_data:np.ndarray,
                     f_data:np.ndarray, alpha_list:list[np.ndarray], nugget=1e-10) -> tuple[np.ndarray, np.ndarray]:
    """    Perform kernel smoothing on the data and estimate multi-derivatives.



    Return the smoothed data and the estimated derivatives. Each grid point
    will contribute one vector. u_smoothed is n_data x n_grid, and
    multi_derivatives is n_data x n_grid x n_alpha.

    Args:
        kernel (Kernel): The kernel function to use
        x_grid (np.ndarray): Grid points
        u_data (np.ndarray): Training data. Values of the solution at the grid points.
        f_data (np.ndarray): Training data. Output values f(x).
        alpha_list[np.ndarray]: List of alpha values for the multi-derivatives to calculate
        nugget (float): Regularization parameter. Default is 1e-10.

    Returns:
        u_smoothed (np.ndarray): Smoothed data
        multi_derivatives (np.ndarray): Estimated multi-derivatives
    """

    if x_grid.ndim == 2:
        print("Reshaping x_grid in smoother")
        x_grid.reshape(-1, 1)

    n_grid_pts = x_grid.shape[-2]
    print("n_grid_pts: ",     n_grid_pts)
    n_pts_per_grid = u_data.shape[-1]
    print("n_pts_per_grid: ", n_pts_per_grid)
    n_alpha = len(alpha_list)

    # Calculate the kernel matrix K using the provided kernel object
    K = np.zeros((n_grid_pts, n_grid_pts))
    for i in range(n_grid_pts):
        for j in range(n_grid_pts):
            K[i, j] = kernel(x_grid[i], u_data[j])

    # Regularize and invert the kernel matrix
    K_reg = K + nugget * np.eye(n_grid_pts)

    # Compute smoothed values u_smoothed using the regularized kernel matrix
    u_smoothed = np.dot(K_reg, u_data)

    # multi_derivatives_list = []
    u_derivatives_list = []
    # Compute multi-derivatives for each alpha in alpha_list
    # First, fix an alpha from the list of multi-derivatives.
    for idx, alpha in enumerate(alpha_list):

        # Initialize the array for multi-derivatives
        multi_derivatives = np.zeros(u_data.shape)
        # multi_derivatives
        for i in range(x_grid.shape[0]):
            # Pass in a matrix of n_points_per_grid x point dimension and a vector of u values
            us = kernel.multiDerivative(x_grid[i], u_data[i], alpha)
            multi_derivatives[i,:] = us
        u_derivatives_list.append(
multi_derivatives @
        )
        # multi_derivatives_list.append(multi_derivatives)

    return u_smoothed, multi_derivatives_list

    #raise NotImplementedError("kernel_smoothing not implemented")


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
    Returns:
        np.ndarray: A 3D tensor of features, of shape (n_data, n_grid, n_features).
    """
    # Extract dimensions
    n_data, n_grid = u_smoothed.shape
    n_features = len(function_list)

    # Initialize the feature tensor
    features = np.zeros((n_data, n_grid, n_features))

    # Iterate over each function to generate features
    for feature_index, func in enumerate(function_list):
        # Apply the function across all data and grid points
        for i in range(n_data):
            for j in range(n_grid):
                # Extract the smoothed value and its derivatives at the current point
                u_val = u_smoothed[i, j]
                derivatives_val = multi_derivatives[i, j]

                # Compute the feature using the current function
                features[i, j, feature_index] = func(u_val, derivatives_val)

    return features

    #raise NotImplementedError("assemble_features not implemented")


def learn_DE_form(kernel:Kernel, s_features:np.ndarray, f_labels:np.ndarray, nugget:float) -> Callable:
    """    Learn the operator from the features and labels.

    Use the kernel method to learn a representation of the operator. Return a function
    that can be used to evaluate the operator at new points.

    Args:
        kernel (Kernel): _description_
        s_features (np.ndarray): _description_
        f_labels (np.ndarray): _description_
        nugget (float): _description_
	Returns:
        function: A function that takes new data points and returns predicted values.
    """

    # Calculate the kernel matrix from the features using the provided kernel object
    K = kernel.matrix(s_features)

    # Regularize the kernel matrix
    K += nugget * np.eye(K.shape[0])

    # Solve for the weights in the kernel space
    # Weights here are alpha in the ridge regression formula: (K + nugget*I)^-1 * Y
    weights = np.linalg.solve(K, f_labels)

    # Return a function that can use these weights to make predictions with new data
    def predictor(new_features):
        # Compute the kernel between the new features and the training features
        k_new = np.array([kernel(new_feature, feature) for new_feature in new_features for feature in s_features]).reshape(len(new_features), len(s_features))

        # Return the predicted values
        return k_new @ weights

    return predictor

    #raise NotImplementedError("learn_operator not implemented")


def kernelized_DE_fit(kernel:Kernel, DE_operator:Callable, u_data:np.ndarray,
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
        # Number of data points
    n_data = len(u_data)

    # Compute the kernel matrix using the provided kernel
    K = np.zeros((n_data, n_data))
    for i in range(n_data):
        for j in range(n_data):
            K[i, j] = kernel(u_data[i], u_data[j])

    # Regularize the kernel matrix
    K += nugget_f * np.eye(n_data)

    # Compute operator values for each data point using the DE_operator
    operator_values = np.array([DE_operator(u) for u in u_data])

    # Solve the system to find weights that map operator values to output data
    # Here we assume that the DE operator directly gives us a form that can be used with the kernel matrix
    weights = np.linalg.solve(K, f_data - operator_values)

    return weights


    #raise NotImplementedError("kernelized_DE_fit not implemented")
