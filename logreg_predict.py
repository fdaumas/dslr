import sys
import copy
import numpy as np
import pandas as pd

from tools import add_intercept, sigmoid
from getData import get_data
from describe import describe
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
        if not isinstance(x, np.ndarray):
            print("x is not a np.ndarray")
        return None
    if theta.size == 0 or x.size == 0:
        print("x or theta is empty")
        return None
    if theta.shape[0] != x.shape[1] + 1:
        print("theta.shape[0] + 1 != x.shape[1]")
        return None

    x_prime = add_intercept(x)
    # print(blue + f"x_prime in logreg_predict: {x_prime}" + reset)
    y_hat = sigmoid(np.matmul(x_prime, theta))
    # print(blue + f"y_hat in logreg_predict: {y_hat}" + reset)
    return y_hat


def from_y_hats_to_house(y_hat_GorH, y_hat_GorS, df):
    if y_hat_GorH is None or y_hat_GorS is None:
        print("error in logreg_predict")
        return None
    result = pd.DataFrame(columns=[
        'Index',
        'is_Gryffindor_or_Slytherin',
        'is_Gryffindor_or_Hufflepuff',
        'Hogwarts House'
    ])
    result['Index'] = df['Index']
    result['is_Gryffindor_or_Slytherin'] = y_hat_GorS
    result['is_Gryffindor_or_Hufflepuff'] = y_hat_GorH
    for i in result['Index']:
        if result.loc[i, 'is_Gryffindor_or_Hufflepuff'] >= 0.5:
            if result.loc[i, 'is_Gryffindor_or_Slytherin'] >= 0.5:
                result.loc[i, 'Hogwarts House'] = 'Gryffindor'
            else:
                result.loc[i, 'Hogwarts House'] = 'Hufflepuff'
        else:
            if result.loc[i, 'is_Gryffindor_or_Slytherin'] >= 0.5:
                result.loc[i, 'Hogwarts House'] = 'Slytherin'
            else:
                result.loc[i, 'Hogwarts House'] = 'Ravenclaw'
    result.to_csv('houses.csv', index=False)
    return result


def from_theta_csv_to_np(file_theta):
    thetas = get_data(file_theta)
    theta_GorH = thetas.loc[0, 'theta_G_or_H']
    theta_GorH = theta_GorH.replace('[', '')
    theta_GorH = theta_GorH.replace(']', '')
    split = theta_GorH.split('\n')
    theta_GorH = []
    for s in split:
        theta_GorH.append(float(s))
    theta_GorH = np.array(theta_GorH).reshape(-1, 1)

    theta_GorS = thetas.loc[0, 'theta_G_or_S']
    theta_GorS = theta_GorS.replace('[', '')
    theta_GorS = theta_GorS.replace(']', '')
    split = theta_GorS.split('\n')
    theta_GorS = []
    for s in split:
        theta_GorS.append(float(s))
    theta_GorS = np.array(theta_GorS).reshape(-1, 1)
    return theta_GorH, theta_GorS


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 3:
        print("give 2 .csv file name as argument")
        exit()
    if argv[1][-4:] != ".csv" or argv[2][-4:] != ".csv":
        print("give .csv file name as argument")
        exit()
    if argv[1][-4:] == ".csv":
        file_test = argv[1]
        file_theta = argv[2]
    elif argv[2][-4:] == ".csv":
        file_test = argv[2]
        file_theta = argv[1]
    else:
        print("give a train.csv file name as argument and other containing theta")
        exit()
    df = get_data(file_test)
    # keep only Herbology and Ancient Runes columns and index
    # need to keep the index because of dropna
    df = df[['Index', 'Herbology', 'Ancient Runes']]
    df = df.dropna()
    # copy for keep index after training
    copy_df = copy.deepcopy(df)

    # no need to keep the index to train or predict
    df = df.drop(columns=['Index'])
    describe_df = describe(df, ['Ancient Runes', 'Herbology'])
    for col in df:
        df[[col]] = (
            df[[col]] - describe_df[col]['min']) / (
            describe_df[col]['max'] - describe_df[col]['min']
        )

    x = df.to_numpy()

    theta_GorH, theta_GorS = from_theta_csv_to_np(file_theta)

    # get predictions
    y_hat_GorH = logreg_predict(x, theta_GorH)
    if y_hat_GorH is None:
        print("error in logreg_predict")
        exit()
    y_hat_GorS = logreg_predict(x, theta_GorS)
    if y_hat_GorS is None:
        print("error in logreg_predict")
        exit()

    # from prediction to Hogwarts House
    result = from_y_hats_to_house(y_hat_GorH, y_hat_GorS, copy_df)

    print(result[['Index', 'Hogwarts House']])
    result = result[['Index', 'Hogwarts House']]
    print(result)
    result.to_csv('houses.csv', index=False)
