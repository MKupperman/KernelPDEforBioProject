import numpy as np
import pytest
from KBio.kernels import Polynomial, Gaussian

# Test cases for gradientX and gradientY methods

# Test cases for Polynomial kernel
def test_polynomial_gradient():
    poly_kernel = Polynomial(degree=2, c=1)
    x = np.array([1, 2, 3])
    y = np.array([2, 3, 4])
    expected_gradient_x = np.array([18, 27, 36])
    expected_gradient_y = np.array([12, 18, 24])

    assert np.allclose(poly_kernel.gradientX(x, y), expected_gradient_x)
    assert np.allclose(poly_kernel.gradientY(x, y), expected_gradient_y)

# Test cases for Gaussian kernel
def test_gaussian_gradient():
    sigma = 1.5
    gaussian_kernel = Gaussian(sigma)
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 3.0, 4.0])
    expected_gradient_x = np.array([-0.15915494, -0.31830989, -0.47746483])
    expected_gradient_y = np.array([0.15915494, 0.31830989, 0.47746483])

    assert np.allclose(gaussian_kernel.gradientX(x, y), expected_gradient_x)
    assert np.allclose(gaussian_kernel.gradientY(x, y), expected_gradient_y)

# Test cases for Polynomial kernel's multiDerivative method
def test_polynomial_multi_derivative():
    poly_kernel = Polynomial(degree=2, c=1)
    x = np.array([1, 2, 3])
    y = np.array([2, 3, 4])
    alpha = [2, 1, 0]  # Second derivative w.r.t. the first component and first derivative w.r.t. the second component
    expected_result = np.array([6, 12, 24])

    result = poly_kernel.multiDerivative(x, y, alpha)

    assert np.allclose(result, expected_result)

# Run the tests
if __name__ == "__main__":
    pytest.main()
