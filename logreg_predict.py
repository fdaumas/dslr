import sys
import numpy as np

from tools import add_intercept, sigmoid
from getData import get_data


# from bootcamp machine learning module 08 ex03
def logreg_predict(x, theta):
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
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        print("x or theta not a np.ndarray")
        return None
    if theta.size == 0 or x.size == 0:
        print("x or theta is empty")
        return None
    if theta.shape[0] != x.shape[1] + 1:
        print("theta.shape[0] + 1 != x.shape[1]")
        return None
    x_prime = add_intercept(x)
    y_hat = sigmoid(np.matmul(x_prime, theta))
    return y_hat


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        print("give .csv file name as argument")
        exit()
    file_name = argv[1]
    if file_name[-4:] != ".csv":
        print("give .csv file name as argument")
        exit()
    df = get_data(file_name)
    # keep only Herbology and Ancient Runes columns and index
    df = df[['Index', 'Herbology', 'Ancient Runes']]
    # drop rows with NaN values
    df = df.dropna()
    # save index
    index = df['Index']
    # drop index column
    df = df.drop(columns=['Index'])
    # get numpy array from df
    x = df.to_numpy()
    # get theta from file
    theta = np.array([[-1.], [0.01], [0.01]]) # ?
    # get predictions
    y_hat = logreg_predict(x, theta)
    print(y_hat)
