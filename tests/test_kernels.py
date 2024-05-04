import numpy as np
import pytest
from KBio.kernels import Polynomial, Gaussian

# # Test cases for gradientX and gradientY methods

# # Test cases for Polynomial kernel
# def test_polynomial_gradient():
#     poly_kernel = Polynomial(degree=2, c=1)
#     x = np.array([1, 2, 3])
#     y = np.array([2, 3, 4])
#     expected_gradient_x = np.array([18, 27, 36])
#     expected_gradient_y = np.array([12, 18, 24])

#     assert np.allclose(poly_kernel.gradientX(x, y), expected_gradient_x)
#     assert np.allclose(poly_kernel.gradientY(x, y), expected_gradient_y)

# # Test cases for Gaussian kernel
# def test_gaussian_gradient():
#     sigma = 1.5
#     gaussian_kernel = Gaussian(sigma)
#     x = np.array([1.0, 2.0, 3.0])
#     y = np.array([2.0, 3.0, 4.0])
#     expected_gradient_x = np.array([-0.15915494, -0.31830989, -0.47746483])
#     expected_gradient_y = np.array([0.15915494, 0.31830989, 0.47746483])

#     assert np.allclose(gaussian_kernel.gradientX(x, y), expected_gradient_x)
#     assert np.allclose(gaussian_kernel.gradientY(x, y), expected_gradient_y)

# # Test cases for Polynomial kernel's multiDerivative method
# def test_polynomial_multi_derivative():
#     poly_kernel = Polynomial(degree=2, c=1)
#     x = np.array([1, 2, 3])
#     y = np.array([2, 3, 4])
#     alpha = [2, 1, 0]  # Second derivative w.r.t. the first component and first derivative w.r.t. the second component
#     expected_result = np.array([6, 12, 24])

#     result = poly_kernel.multiDerivative(x, y, alpha)

#     assert np.allclose(result, expected_result)

# # Run the tests
# if __name__ == "__main__":
#     pytest.main()


# import numpy as np
# import pytest
# from KBio.kernels import Polynomial  # Ensure this path matches your project structure

def test_polynomial_kernel_gradient():
    # Parameters for the Polynomial kernel
    degree = 2
    constant = 12
    kernel = Polynomial(degree=degree, c=constant)

    # Test inputs as scalars, but the operation may expect arrays
    x = np.array([2])  # Use numpy array to support vector operations
    y = np.array([3])

    # Manually calculated expected gradient
    expected_gradient = degree * (x * y + constant) ** (degree - 1) * y

    # Compute the actual gradient using the kernel's method
    actual_gradient = kernel.gradientX(x, y)

    # Use pytest to assert that the expected and actual gradients are close
    np.testing.assert_array_almost_equal(actual_gradient, expected_gradient, decimal=5,
                                         err_msg="Gradient of Polynomial kernel is incorrect.")


def test_polynomial_kernel_multiderivative_cubic():
    # Define a simple polynomial kernel
    degree = 3
    constant = 0
    kernel = Polynomial(degree=degree, c=constant)

    alpha = [1]  # First derivative with respect to the first component of x
    # Test case setup
    for x in np.asarray([1e-1, 1e-2, 2e-1, 0.5, 0.73, 2,3,4,5,6,7,8, 20, 50, 100]):
        for y in np.asarray([1e-1, 1e-2, 2e-1, 0.5, 0.73, 2,3,4,5,6,7,8,  20, 50, 100]):
            x = np.array([x])
            y = np.array([y])

            # Manual calculation of the derivative:
            # For the first derivative of K(x, y) = (x * y + 1)^3,
            # the derivative dK/dx = 3 * (x * y + 1)^2 * y
            expected_derivative = 3 * (x[0] * y[0] + constant)**2 * y[0]

            # Get the actual derivative from the kernel function
            actual_derivative = kernel.multiDerivative(x, y, alpha)

            # Verify the actual derivative is close to the expected
            err_msg = f"Multiderivative of Polynomial kernel is incorrect at {x=}, {y=}. Err = {actual_derivative - expected_derivative}"
            atol = 0
            rtol = 1e-8
            print(np.max(np.abs(actual_derivative - expected_derivative)))
            assert np.max(np.abs(actual_derivative - expected_derivative)) < atol + rtol * np.max(np.abs(expected_derivative)), err_msg

def test_polynomial_kernel_multiderivative():
    # Define a simple polynomial kernel
    degree = 3
    constant = 1
    kernel = Polynomial(degree=degree, c=constant)

    alpha = [1]  # First derivative with respect to the first component of x
    # Test case setup
    for x in np.asarray([1e-1, 1e-2, 2e-1, 0.5, 0.73, 2,3,4,5,6,7,8]):
        for y in np.asarray([1e-1, 1e-2, 2e-1, 0.5, 0.73, 2,3,4,5,6,7,8]):
            x = np.array([x])
            y = np.array([y])

            # Manual calculation of the derivative:
            # For the first derivative of K(x, y) = (x * y + 1)^3,
            # the derivative dK/dx = 3 * (x * y + 1)^2 * y
            expected_derivative = 3 * (x[0] * y[0] + constant)**2 * y[0]

            # Get the actual derivative from the kernel function
            actual_derivative = kernel.multiDerivative(x, y, alpha)

            # Verify the actual derivative is close to the expected
            err_msg = f"Multiderivative of Polynomial kernel is incorrect at {x=}, {y=}. Err = {actual_derivative - expected_derivative}"
            atol = 0
            rtol = 1e-5
            print(np.max(np.abs(actual_derivative - expected_derivative)))
            assert np.max(np.abs(actual_derivative - expected_derivative)) < atol + rtol * np.max(np.abs(expected_derivative)), err_msg
