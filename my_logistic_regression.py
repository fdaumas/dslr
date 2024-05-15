import numpy as np
from math import exp


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        if (isinstance(theta, list)):
            self.theta = np.array(theta)
            if (self.theta.shape != (len(self.theta), 1)):
                self.theta = self.theta.reshape((len(self.theta), 1))
        else:
            self.theta = theta

    def add_intercept(self, x):
        """Adds a column of 1's to the non-empty numpy.array x.
        Args:
            to be a numpy.array of dimension m * n.
        Returns:
            X, a numpy.array of dimension m * (n + 1).
            None if x is not a numpy.array.
            None if x is an empty numpy.array.
        Raises:
            function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or x.size == 0:
            print("x not a np.ndarray or x.size = 0")
            return None
        if x.ndim == 1:
            res = []
            nb_line = x.shape[0]
            for i in range(nb_line):
                res.append(1)
                res.append(x[i])
            res = np.array(res)
            res = res.reshape(nb_line, 2)
            return res

        col1 = list()
        for i in range(x.shape[0]):
            col1.append(1)
        res = np.insert(x, 0, col1, axis=1)
        return res
    
    def sigmoid_(self, x):
        """
        Compute the sigmoid of a vector.
        Args:
                x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
                The sigmoid value as a numpy.ndarray of shape (m, 1).
                None if x is an empty numpy.ndarray.
        Raises:
                This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray):
            return None
        if x.ndim == 2 and x.shape[1] != 1:
            return None
        res = np.ones(x.size).reshape(x.shape)
        for i in range(x.shape[0]):
            res[i][0] = 1 / (1 + exp(-x[i][0]))
        return res
    
    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray
        Args:
                x: has to be an numpy.ndarray, a vector of dimension m * n.
                theta: has to be an numpy.ndarray, 
                a vector of dimension (n + 1) * 1.
        Returns:
                y_hat as a numpy.ndarray, a vector of dimension m * 1.
                None if x or theta are empty numpy.ndarray.
                None if x or theta dimensions are not appropriate.
        Raises:
                This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray):
            print("x or theta not a np.ndarray")
            return None
        if self.theta.size == 0 or x.size == 0:
            print("x or theta is empty")
            return None
        if self.theta.shape[0] != x.shape[1] + 1:
            print("theta.shape[0] + 1 != x.shape[1]")
            return None
        x_prime = self.add_intercept(x)
        y_hat = self.sigmoid_(np.matmul(x_prime, self.theta))
        return y_hat
    
    def gradient(self, x, y):
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
            or not isinstance(self.theta, np.ndarray)
        ):
            return None
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            return None
        if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.theta.shape[0]:
            return None
        x_prime_T = np.transpose(self.add_intercept(x))
        m = x.shape[0]
        y_hat = self.predict_(x)
        res = 1 / m * np.matmul(x_prime_T, y_hat - y)
        return res
    
    def fit_(self, x, y):
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
            or not isinstance(self.theta, np.ndarray)
            or not isinstance(self.alpha, float)
            or not isinstance(self.max_iter, int)
        ):
            print("x or y or theta not a np.ndarray or alpha not a float or max_iter not an int")
            return None
        if x.ndim != 2 or y.ndim != 2 or self.theta.ndim != 2:
            print("x.ndim != 2 or y.ndim != 2 or self.theta.ndim != 2")
            return None
        if self.alpha > 1 or self.alpha < 0 or self.max_iter < 1:
            print("alpha too big or negative or max_iter <= 0")
            return None
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            print("x or y or theta is empty")
            return None
        if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.theta.shape[0]:
            print("x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]")
            return None      
        
        gradient = np.ndarray(y.shape)
        for i in range(self.max_iter):
            gradient = self.gradient(x, y)
            if (isinstance(gradient, str)):
                return "error"
            # if (i > self.max_iter -10):
            #     print(f"g = {gradient[1]}")
            self.theta = self.theta - self.alpha * gradient
            # if (i % 10000 == 0):
            #     print(f"i = {i} et theta = {self.theta}")
        return gradient[0][0]
    
    def loss_(self, y, y_hat):
        """
        Compute the logistic loss value.
        Args:
                y: has to be an numpy.ndarray, a vector of shape m * 1.
                y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
                The logistic loss value as a float.
                None on any error.
        Raises:
                This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            print("y or y_hat is not a np.ndarray")
            return None
        
        if y.shape != y_hat.shape:
            print("y and y_hat doesn't have the same shape")
            return None
        eps = 1e-15
        ones = np.ones(y.size).reshape(y.shape)
        diff = ones - y_hat
        for i in range(y_hat.shape[0]):
            if y_hat[i][0] == 0:
                y_hat[i][0] = eps
            if diff[i][0] == 0:
                diff[i][0] = eps
        log1 = np.log(y_hat)
        log2 = np.log(diff)
        m = y.shape[0]
        dot1 = np.matmul(np.transpose(y), log1)
        dot2 = np.matmul(np.transpose(ones - y), log2)
        res = 1 / -m * (dot1 + dot2)
        return res[0][0]