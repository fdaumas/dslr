import sys
import numpy as np
import pandas as pd

from tools import add_intercept, sigmoid
from getData import get_data
from color import red, green, yellow, blue, reset


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
    print(blue + f"x_prime in logreg_predict: {x_prime}" + reset)
    y_hat = sigmoid(np.matmul(x_prime, theta))
    print(blue + f"y_hat in logreg_predict: {y_hat}" + reset)
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
    # need to keep the index because of dropna
    df = df[['Index', 'Herbology', 'Ancient Runes']]
    df = df.dropna()

    # prepare a df of results
    result = pd.DataFrame(columns=[
        'Index',  # need to keep the index because of dropna
        'is_Gryffindor_or_Slytherin',
        'is_Gryffindor_or_Hufflepuff',
        'Hogwarts House'
    ])

    # save the index
    result['Index'] = df['Index']

    # no need to keep the index to train or predict
    df = df.drop(columns=['Index'])

    x = df.to_numpy()

    # get theta from train
    theta_GorH = np.array([[-1.], [0.01], [0.01]])
    theta_GorS = np.array([[-1.], [0.01], [0.01]])

    # get predictions
    y_hat_GorH = logreg_predict(x, theta_GorH)
    if y_hat_GorH is None:
        print("error in logreg_predict")
        exit()
    y_hat_GorS = logreg_predict(x, theta_GorS)

    # from prediction to Hogwarts House
    result['is_Gryffindor_or_Slytherin'] = y_hat_GorS
    result['is_Gryffindor_or_Hufflepuff'] = y_hat_GorH

    result.loc[
        result['is_Gryffindor_or_Hufflepuff'] < 0.5
        and result['is_Gryffindor_or_Slytherin'] >= 0.5,
        'Hogwarts House'
    ] = 'Slytherin'
    result.loc[
        result['is_Gryffindor_or_Hufflepuff'] >= 0.5
        and result['is_Gryffindor_or_Slytherin'] < 0.5,
        'Hogwarts House'
    ] = 'Hufflepuff'
    result.loc[
        result['is_Gryffindor_or_Hufflepuff'] >= 0.5
        and result['is_Gryffindor_or_Slytherin'] >= 0.5,
        'Hogwarts House'
    ] = 'Gryffindor'
    result.loc[
        result['is_Gryffindor_or_Hufflepuff'] < 0.5
        and result['is_Gryffindor_or_Slytherin'] < 0.5,
        'Hogwarts House'
    ] = 'Ravenclaw'
    print(result[['Index', 'Hogwarts House']])
