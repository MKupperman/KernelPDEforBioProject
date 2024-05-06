from typing import Callable

import numpy as np
from .kernels import Kernel
from .optim import regularized_cholesky_solve

def kernel_smoothing(kernel:Kernel, x_grid:np.ndarray, u_data:np.ndarray,
                     f_data:np.ndarray, alpha_list:list[np.ndarray], nugget=1e-3) -> tuple[np.ndarray, np.ndarray]:
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
        x_grid = x_grid[:,:, np.newaxis]
        print(x_grid.shape)

    n_grid_pts = x_grid.shape[-2]
    print("n_grid_pts: ",     n_grid_pts)

    # First, we calculate the kernel matrix for the grid points of the reference data
    K = kernel.matrix(x_grid[0])

    # Solve the "Core" problem outside of the loop,
    # e.g. z = (U(X,X) + lam2*I)^-1 * u(X)
    z = regularized_cholesky_solve(K, u_data, lam2=nugget)

    # Allocate space for the smoothed data
    u_derivatives_list = []
    kernel_derivatives_list = []

    for alpha in alpha_list:
        # NOTE - if the grid we want to smooth onto is different than the training,
        # provide a different x argument.
        # We want (n_smoothing_in, n_reference_points) matrix shape derivative matrix
        Ks = kernel.multiDerivative(x=x_grid[0], y=x_grid[0], alpha=alpha)
        kernel_derivatives_list.append(Ks)
        u_smoothed = Ks.dot(z)
        u_derivatives_list.append(u_smoothed)
        # multi_derivatives_list.append(multi_derivatives)

    return z, kernel_derivatives_list, u_derivatives_list


def assemble_features(x_grid_list, u_smoothed, multi_derivatives, function_list,
                      flatten=True)-> np.ndarray:
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
            functions should have the call signature f(x, u, multi_derivatives), and return
            a scalar.
    Returns:
        np.ndarray: A 3D tensor of features, of shape (n_data, n_grid, n_features).
    """
    # Extract dimensions
    n_grid, n_data = u_smoothed.shape
    n_derivatives = len(multi_derivatives) + 1
    n_features = len(function_list) + len(x_grid_list[0]) + n_derivatives

    # Initialize the feature tensor
    features = np.zeros((n_data, n_grid, n_features))
    # First, select which series we are going to use
    for i in range(n_data):
        # select the j'th grid point
        for j in range(n_grid):
            x = x_grid_list[j]
            u_val = u_smoothed[j,i]
            u_derivs = [uprime[j,i] for uprime in multi_derivatives]
            f_list = [f(x, u_val, u_derivs) for f in function_list]

            features[i,j,:] = np.hstack([np.asarray(x), np.asarray(u_val), np.asarray(u_derivs), np.asarray(f_list)])
    if flatten:
        return features.reshape(features.shape[0] * features.shape[1], features.shape[2])
    else:
        return features


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

    # TODO Replace this with a cholesky solve
    weights = np.linalg.solve(K, f_labels)

    # Return a function that can use these weights to make predictions with new data
    def predictor(new_features):
        """   Predict values using the learned operator.

        Args:
            new_features (np.ndarray): New data points to predict values for. Shape is (n_data_new, n_features).
        """
        # Compute the kernel between the new features and the training features
        # Convert lists to numpy arrays
        new_features_np = np.array(new_features)    # (n_data_new, n_features)
        s_features_np = np.array(s_features)        # (n_data, n_features)


        # Apply kernel function to each pair of new_feature and feature
        # TODO Re-write this if/when it's vectorizable.
        k_new = np.ones((new_features_np.shape[0], s_features_np.shape[0]))
        for i in range(new_features_np.shape[0]):
            for j in range(s_features_np.shape[0]):
                k_new[i, j] = kernel(new_features_np[i], s_features_np[j])
        print(k_new.shape, weights.shape)
        # This is neat, but it's difficult to understand what's going on
        # k_new = np.array([kernel(new_feature, feature) for new_feature in new_features for feature in s_features]).reshape(len(new_features), len(s_features))

        # Return the predicted values
        return k_new, k_new @ weights

    return predictor, weights

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
