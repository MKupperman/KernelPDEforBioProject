import numpy as np
from scipy.special import factorial, hermite
from itertools import product
from abc import ABC, abstractmethod
# import Hermite Hn function from scipy
from scipy.special import eval_hermite as hermite

# Create an ABC for kernel functions. This is useful for creating a common interface for all kernel functions.

class Kernel(ABC):
    @abstractmethod
    def __call__(self, x, y) -> float:
        """The kernel function, K(x,y) is implemented here.



        Args:
            x : first input
            y : second input

        Returns:
            float : value of the kernel function at x and y
        """

        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def feature_map(self, x, n=1):
        """ Return a feature map representation of the input x.

        Second argument n is to specify dimension of the feature map if it is infinite dimensional.
        If the feature map is always finite dimensional, n is ignored.
        """
        pass

    @abstractmethod
    def matrix(self, X):
        n = len(X)
        K = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self(X[i], X[j])
        return K

    @abstractmethod
    def gradientX(self, x, y):
        pass

    def gradientY(self, x, y):
        """Evaluate the gradient of the kernel function with respect to the
            second argument.

            As kernel functions are symmetric, this is the same as the gradient
            with respect to the first argument. It is provided for completeness.

        Args:
            x (numeric): First argument
            y (numeric): Second argument

        Returns:
            gradient: numeric
        """
        return self.gradientX(y, x)

    @abstractmethod
    def multiDerivative(self, x, y, alpha:list[int]):
        # Compute the alpha-partial derivative of the kernel function w.r.t x.

        # Use JAX here to implement this function, or do it analytically
        raise NotImplementedError("multiDerivative not implemented")

class Linear(Kernel):
    def __call__(self, x, y):
        return np.dot(x, y)

    def __str__(self):
        return 'Linear'

    def gradientX(self, x, y):
        return y

    def gradientY(self, x, y):
        return x

    def matrix(self, X):
        return np.dot(X, X.T)
    def multiDerivative(self, x, y, alpha: list[int]):

        raise NotImplementedError("multiDerivative not implemented")


class Polynomial(Kernel):
    def __init__(self, degree, c=1):
        self.degree = degree
        self.c = c

    def __call__(self, x, y):
        return (np.dot(x, y) + self.c) ** self.degree

    def __str__(self):
        return f'Polynomial degree {self.degree}'

    def feature_map(self, x:np.ndarray, n=1, max_dims=True):
        """

        Args:
            x (np.ndarray): _description_
            n (int, optional): Specify a lower dimension for approximation if
                high degree. Disregarded if `max_dims=True`. Defaults to 1.
            max_dims (bool, optional): Disregard `n` and use the dimension `d`
                for an exact representation. Defaults to True.

        Returns:
            np.ndarray: Feature map of length `n` or `d` depending on `max_dims`.
        """
        max_power = min(self.degree, n)

        # Compute the feature vector components
        features = np.zeros(max_power + 1)

        for k in range(max_power + 1):
            binomial_coeff = comb(self.degree, k)
            feature_component = binomial_coeff * (self.c ** (self.d - k))
            feature_component *=  (np.linalg.norm(x) ** k)
            features[k] = feature_component

        return features

        # raise NotImplementedError("feature_map not implemented")

    # Check the rest of this class - was generated by Copilot
    def gradientX(self, x, y):
        return self.degree * (np.dot(x, y) + self.c) ** (self.degree - 1) * y

    def gradientY(self, x, y):
        return self.degree * (np.dot(x, y) + self.c) ** (self.degree - 1) * x

    def matrix(self, X):
        return (np.dot(X, X.T) + 1) ** self.degree

    def multiDerivative(self, x:np.ndarray, y:np.ndarray, alpha: list[int]):
        """Compute the alpha-partial derivative of the kernel function w.r.t x.

        Given a vectors x and y, compute the alpha-partial derivative of the
        kernel function w.r.t x at the pair (x,y).

        x is either a vector or a matrix of vectors. If x is a matrix, each column

        Alpha is the order of derivatives

        Args:
            x (np.ndarray): input vector
            y (np.ndarray): input vector
            alpha (list[int]): order of derivatives
        """
        alphas = np.asarray(alpha)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)

        d = np.zeros(x.shape[0])
        for yn in y:
            a = np.prod[np.power(yn, alphas)]
            b = comb(self.degree, len(alphas))
            for i, xn in enumerate(x):
                d[i] = a * b (xn.T @ y + self.c)**(self.degree - len(alpha))
        # raise NotImplementedError("multiDerivative not implemented")


class Gaussian(Kernel):
    """ Also called the RBF Kernel."""
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x, y):
        # filter on if x and y are vectors or matrices
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * self.sigma ** 2))

    def __str__(self):
        return f'Gaussian with sigma {self.sigma}'

    def feature_map(self, x, n=1):

        x = np.array(x, dtype=float)
        # Generate n/2 frequencies for cosine and sine components
        frequencies = np.linspace(1, n//2, n//2) / (2 * np.pi * self.sigma**2)

        # Calculate the feature vector components using both cosine and sine
        features_cos = np.cos(frequencies[:, None] * x[None, :]).flatten()
        features_sin = np.sin(frequencies[:, None] * x[None, :]).flatten()

        # Combine the features and scale them
        features = np.sqrt(1 / n) * np.concatenate([features_cos, features_sin])
        return features


    def gradientX(self, x, y):
        factor = -1 / self.sigma ** 2
        return factor * (x - y) * self.__call__(x, y)

    def matrix(self, X):
        # Check this:
        K = np.exp(-np.linalg.norm(X[:, None] - X[None, :], axis=2) ** 2 / (2 * self.sigma ** 2))
        # n = len(X)
        # K = np.zeros((n, n))
        # for i in range(n):
        #     for j in range(i, n):
        #         K[i, j] = self.__call__(X[i], X[j])
        #         K[j, i] = K[i, j]
        return K

    '''
    def multiDerivative(self, x, y, alpha: list[int]):
        Kxy = self.__call__(x, y)
        if len(alpha) == 1 and alpha[0] == 1:
            return self.gradientX(x, y)
        elif len(alpha) == 2 and all(a == 1 for a in alpha):
            factor = 1 / self.sigma ** 4
            term1 = self.sigma ** 2 - np.sum((x - y) ** 2)
            return factor * (x - y) ** 2 * Kxy + term1 * Kxy / self.sigma ** 2
        else:
            raise NotImplementedError("Higher order derivatives are not implemented")
    '''
    def multiDerivative(self, x, y, alpha: list[int]):
        n = sum(alpha)
        if n == 0:
            return self.__call__(x, y)

        Kxy = self.__call__(x, y)
        factor = (-1 / self.sigma ** 2) ** n
        derivatives = np.zeros_like(x)

        # Calculate each component's contribution
        for i, a in enumerate(alpha):
            if a > 0:
                Hn = hermite(a)  # Get the Hermite polynomial of order a
                xi = (x[i] - y[i]) / self.sigma
                derivatives[i] = Hn(xi) * np.exp(-xi ** 2 / 2)  # Evaluate the Hermite polynomial at xi

        # Combine all components
        total_derivative = np.prod(derivatives)
        return factor * total_derivative * Kxy

        '''
        gaussian_kernel = Gaussian(sigma=1.5)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.5, 2.5, 3.5])
        alpha = [2, 1, 0]  # Second derivative w.r.t. the first component and first derivative w.r.t. the second component
        result = gaussian_kernel.multiDerivative(x, y, alpha)
        print("Derivative result:", result)
        '''


class Exponential(Kernel):
    def __init__(self, l):
        self.l = l

    # Class implementation by Copilot - check if it is correct
    def __call__(self, x, y):
        return np.exp(-np.linalg.norm(x - y) / self.l)

    def __str__(self):
        return f'Exponential l {self.l}'

    def feature_map(x, n=1):
        raise NotImplementedError("feature_map not implemented")

    def gradientX(self, x, y):
        return -self(x, y) * (x - y) / self.l ** 2

    def matrix(self, X):
        n = len(X)
        K = np.zeros((n,n))
        for i in range(n):
            for j in range(i, n):
                K[i,j] = self(X[i], X[j])
                K[j,i] = K[i,j]
        return K

    def multiDerivative(self, x, y, alpha: list[int]):

        raise NotImplementedError("multiDerivative not implemented")

class featureMapKernel(Kernel):
    # A kernel can be defined by a feature map and a set of weights.
    # Use this to define a kernel function.

    # Implementation by Copilot - check if it is correct
    def __init__(self, feature_map, weights, alpha_deriv_formula):
        self.feature_map = feature_map
        self.weights = weights
        self.alpha_deriv_formula = alpha_deriv_formula

    def __call__(self, x, y):
        return np.dot(self.feature_map(x), self.feature_map(y))

    def __str__(self):
        return 'Feature Map Kernel'

    def gradientX(self, x, y):
        return self.feature_map(y)

    def matrix(self, X):
        n = len(X)
        K = np.zeros((n,n))
        for i in range(n):
            for j in range(i, n):
                K[i,j] = self(X[i], X[j])
                K[j,i] = K[i,j]
        return K

    def multiDerivative(self, x, y, alpha: list[int]):
        # Use alpha_deriv_formula to compute the alpha-partial derivative of the kernel function w.r.t x.
        if sum(alpha) == 1:  # First-order derivative
            # Assuming alpha corresponds to the order of derivative w.r.t. each component.
            result = self.degree * (np.dot(x, y) + 1) ** (self.degree - 1)
            # Multiply by the derivative of the inner product w.r.t. the respective component.
            grad = []
            for i, a in enumerate(alpha):
                if a == 1:
                    grad.append(y[i] if i < len(y) else 0)
            return result * np.array(grad)
        else:
            raise NotImplementedError("Higher or zero order derivatives are not implemented")