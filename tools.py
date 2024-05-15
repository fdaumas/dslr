import numpy as np
from math import exp


def sigmoid(x):
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


def add_intercept(x):
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
		return None
	if (x.ndim == 1):
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
	res = np.insert(x, 0, col1, axis = 1)
	return res
