import numpy as np

from logreg_predict import logreg_predict


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without
    any for-loop. The three arrays must have comp
    Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
            The gradient as a numpy.ndarray, a vector of shape n * 1, containg
            the result of the formula for all j.
            None if x, y, or theta are empty numpy.ndarray.
            None if x, y and theta do not have compatible shapes.
    Raises:
            This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(theta, np.ndarray)
    ):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None
    x_prime_T = np.transpose(add_intercept(x))
    m = x.shape[0]
    y_hat = logreg_predict(x)
    res = 1 / m * np.matmul(x_prime_T, y_hat - y)
    return res


def fit_(theta, alpha, max_iter, x, y):
    """
    Description:
            Fits the model to the training dataset contained in x and y.
    Args:
            x: has to be a numpy.array, a matrix of dimension m * n:
            (number of training examples, number of features).
            y: has to be a numpy.array, a vector of dimension m * 1:
            (number of training examples, 1).
            theta: has to be a numpy.array, a vector
            of dimension (n + 1) * 1: (number of features + 1, 1).
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done
            during the gradient descent
    Return:
            new_theta: numpy.array, a vector of dimension
            (number of features + 1, 1).
            None if there is a matching dimension problem.
            None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
            This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or not isinstance(alpha, float)
        or not isinstance(max_iter, int)
    ):
        print("x or y or theta not a np.ndarray or alpha not a float or max_iter not an int")
        return None
    if x.ndim != 2 or y.ndim != 2 or theta.ndim != 2:
        print("x.ndim != 2 or y.ndim != 2 or self.theta.ndim != 2")
        return None
    if alpha > 1 or alpha < 0 or max_iter < 1:
        print("alpha too big or negative or max_iter <= 0")
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        print("x or y or theta is empty")
        return None
    if x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        print("x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]")
        return None

    gradient = np.ndarray(y.shape)
    for i in range(max_iter):
        gradient = gradient(x, y)
        if (isinstance(gradient, str)):
            return "error"
        # if (i > self.max_iter -10):
        #     print(f"g = {gradient[1]}")
        theta = theta - alpha * gradient
        # if (i % 10000 == 0):
        #     print(f"i = {i} et theta = {self.theta}")
    return theta
