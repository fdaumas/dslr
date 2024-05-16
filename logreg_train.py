import numpy as np
import copy
import pandas as pd

from logreg_predict import logreg_predict
from getData import get_data
from describe import describe
from tools import df_to_is_G_or_H, df_to_is_G_or_S, add_intercept
from tools import df_to_is_G, df_to_is_S, df_to_is_H, df_to_is_R
from color import red, green, yellow, blue, reset, bold


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
    # print(green + f"x_prime_T in gradienr = {x_prime_T}" + reset)
    m = x.shape[0]
    # print(green + f"m in gradient = {m}" + reset)
    y_hat = logreg_predict(x, theta)
    # print(green + f"y_hat in gradient = {y_hat}" + reset)
    res_tmp = np.matmul(x_prime_T, y_hat - y)
    # print(green + f"res_tmp in gradient = {res_tmp}" + reset)
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
    if alpha > 1000 or alpha < 0 or max_iter < 1:
        print("alpha too big or negative or max_iter <= 0")
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        print("x or y or theta is empty")
        return None
    if x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        print("x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]")
        return None

    my_gradient = np.ones(y.shape)
    for i in range(max_iter):
        # print(my_gradient)
        my_gradient = gradient(x, y, theta)
        # print(yellow + f"my_gradient in fit = {my_gradient}" + reset)
        if (isinstance(my_gradient, str)):
            return "error"
        # if (i > self.max_iter -10):
        #     print(f"g = {gradient[1]}")
        theta = theta - alpha * my_gradient
        # print(yellow + f"theta in fit = {theta}" + reset)
        # if (i % 10000 == 0):
        #     print(f"i = {i} et theta = {self.theta}")
    return theta


def logreg_train(df):
    tmp_df = copy.deepcopy(df)
    tmp_df = tmp_df[['Hogwarts House', 'Herbology', 'Ancient Runes']]
    tmp_df = tmp_df.dropna()
    G_df = df_to_is_G(tmp_df)
    S_df = df_to_is_S(tmp_df)
    H_df = df_to_is_H(tmp_df)
    R_df = df_to_is_R(tmp_df)
    # gryffindor_or_hufflepuff_df = df_to_is_G_or_H(tmp_df)
    # gryffindor_or_slytherin_df = df_to_is_G_or_S(tmp_df)
    note_df = copy.deepcopy(tmp_df)
    note_df = note_df[['Ancient Runes', 'Herbology']]
    describe_df = describe(note_df, ['Ancient Runes', 'Herbology'])
    for col in note_df:
        note_df[[col]] = (
            note_df[[col]] - describe_df[col]['min']) / (
            describe_df[col]['max'] - describe_df[col]['min']
        )
    x = note_df.to_numpy()
    # y_G_or_H = gryffindor_or_hufflepuff_df[['is_Gryffindor_or_Hufflepuff']].to_numpy()
    # y_G_or_S = gryffindor_or_slytherin_df[['is_Gryffindor_or_Slytherin']].to_numpy()
    y_G = G_df[['is_Gryffindor']].to_numpy()
    y_S = S_df[['is_Slytherin']].to_numpy()
    y_H = H_df[['is_Hufflepuff']].to_numpy()
    y_R = R_df[['is_Ravenclaw']].to_numpy()
    # print(describe_df)
    # print(gryffindor_or_hufflepuff_df)
    # print(note_df)
    max_iter = 10000 - 1
    learning_rate = 10.0
    print(f"max_iter = {max_iter} and learning_rate = {learning_rate}")
    theta_G = np.array([1, 1, 1]).reshape(-1, 1)
    theta_G = fit_(theta_G, learning_rate, max_iter, x, y_G)
    print(f"theta_G = {theta_G}")
    theta_S = np.array([1, 1, 1]).reshape(-1, 1)
    theta_S = fit_(theta_S, learning_rate, max_iter, x, y_S)
    print(f"theta_S = {theta_S}")
    theta_H = np.array([1, 1, 1]).reshape(-1, 1)
    theta_H = fit_(theta_H, learning_rate, max_iter, x, y_H)
    print(f"theta_H = {theta_H}")
    theta_R = np.array([1, 1, 1]).reshape(-1, 1)
    theta_R = fit_(theta_R, learning_rate, max_iter, x, y_R)
    print(f"theta_R = {theta_R}")

    # theta_G_or_H = np.array([1, 1, 1]).reshape(-1, 1)
    # theta_G_or_H = fit_(theta_G_or_H, learning_rate, max_iter, x, y_G_or_H)
    # theta_G_or_S = np.array([1, 1, 1]).reshape(-1, 1)
    # theta_G_or_S = fit_(theta_G_or_H, learning_rate, max_iter, x, y_G_or_S)
    # print(f"theta_G_or_H = {theta_G_or_H}")
    # print(f"theta_G_or_S = {theta_G_or_S}")

    theta_df = pd.DataFrame(index=[0], columns=['theta_G', 'theta_S', 'theta_H', 'theta_R'])
    theta_df.loc[0, 'theta_G'] = theta_G
    theta_df.loc[0, 'theta_S'] = theta_S
    theta_df.loc[0, 'theta_H'] = theta_H
    theta_df.loc[0, 'theta_R'] = theta_R
    # theta_df.loc[0, 'theta_G_or_H'] = theta_G_or_H
    # theta_df.loc[0, 'theta_G_or_S'] = theta_G_or_S
    theta_df.to_csv('theta.csv')


if __name__ == "__main__":
    df = get_data('datasets/dataset_train.csv')
    # get G or H df
    # get notes_df
    # get y from G or H df
    # get x from notes df
    # init theta
    # init alpha
    # init max_iter
    # launch fit_
    # save theta as theta_G_or_H

    # pareil avev G or S
    # save theta as theta_G_or_S
    # return both thetas
    logreg_train(df)
