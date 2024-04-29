# optim.py
# (c) the authors.
#
# Kernel optimization and root-finding algorithms.

import numpy as np
from .kernels import Kernel
from scipy.linalg import cholesky, solve_triangular

def newton_kernel(f:Kernel, y, x0, tol, h=1):
    """
    Newton's method for root finding of kernel functions.

    Solve the optimization problem:

    minimize f(x, y) with respect to x.

    Args:
        f: Kernel function.
        y: Second argument to kernel function. Used in optimization.
        x0: Initial guess.
        tol: Tolerance for stopping criterion.
        h: Step size for damped Newton's method. Default is 1.

    Returns:
        Root.
    """
    x = x0
    while True:
        x1 = x - h * f(x,y) / f.gradientX(x, y)
        if np.norm(x1 - x) / len(x) < tol:
            return x1
        x = x1

def kernel_smooth(f:Kernel, X, lam2):
    """
    Kernel smoothing.

    Args:
        f: Kernel function.
        X: Data.
        lam: Regularization parameter.

    Returns:
        Smoothed data.
    """
    n = len(X)
    K = f.matrix(X)

    Cholesky_L = cholesky(K + lam2 * np.eye(n))
    U_sol = solve_triangular(Cholesky_L, (Cholesky_L.T).dot(X), lower=True)
    return u_sol

def GaussNewton(f_obj, f_jacobian, x0, tol):
    raise NotImplementedError("GaussNewton not implemented")

def L_BFGS(f,f_grad, f_hessian, x0, tol):
    raise NotImplementedError("L_BFGS not implemented")
