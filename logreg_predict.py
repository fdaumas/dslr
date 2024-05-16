import sys
import copy
import numpy as np
import pandas as pd

from tools import add_intercept, sigmoid
from getData import get_data
from describe import describe
from color import red, green, yellow, blue, reset
from statsTools import min, max


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


def from_y_hats_to_house(y_hat_G, y_hat_S, y_hat_H, y_hat_R, df):
    if y_hat_G is None or y_hat_S is None or y_hat_H is None or y_hat_R is None:
        print("error in logreg_predict")
        return None
    result = pd.DataFrame(columns=[
        'Index',
        'is_Gryffindor',
        'is_Slytherin',
        'is_Hufflepuff',
        'is_Ravenclaw',
        'Hogwarts House'
    ])
    result['Index'] = df['Index']
    result['is_Gryffindor'] = y_hat_G
    result['is_Slytherin'] = y_hat_S
    result['is_Hufflepuff'] = y_hat_H
    result['is_Ravenclaw'] = y_hat_R
    for i in result['Index']:
        if (
            result.loc[i, 'is_Gryffindor'] > result.loc[i, 'is_Slytherin']
            and result.loc[i, 'is_Gryffindor'] > result.loc[i, 'is_Hufflepuff']
            and result.loc[i, 'is_Gryffindor'] > result.loc[i, 'is_Ravenclaw']
        ):
            result.loc[i, 'Hogwarts House'] = 'Gryffindor'
        elif (
            result.loc[i, 'is_Slytherin'] > result.loc[i, 'is_Gryffindor']
            and result.loc[i, 'is_Slytherin'] > result.loc[i, 'is_Hufflepuff']
            and result.loc[i, 'is_Slytherin'] > result.loc[i, 'is_Ravenclaw']
        ):
            result.loc[i, 'Hogwarts House'] = 'Slytherin'
        elif (
            result.loc[i, 'is_Hufflepuff'] > result.loc[i, 'is_Gryffindor']
            and result.loc[i, 'is_Hufflepuff'] > result.loc[i, 'is_Slytherin']
            and result.loc[i, 'is_Hufflepuff'] > result.loc[i, 'is_Ravenclaw']
        ):
            result.loc[i, 'Hogwarts House'] = 'Hufflepuff'
        else:
            result.loc[i, 'Hogwarts House'] = 'Ravenclaw'
    result.to_csv('houses.csv', index=False)
    return result


def from_theta_csv_to_np(file_theta):
    thetas = get_data(file_theta)
    theta_G = thetas.loc[0, 'theta_G']
    theta_G = theta_G.replace('[', '')
    theta_G = theta_G.replace(']', '')
    split = theta_G.split('\n')
    theta_G = []
    for s in split:
        theta_G.append(float(s))
    theta_G = np.array(theta_G).reshape(-1, 1)

    theta_S = thetas.loc[0, 'theta_S']
    theta_S = theta_S.replace('[', '')
    theta_S = theta_S.replace(']', '')
    split = theta_S.split('\n')
    theta_S = []
    for s in split:
        theta_S.append(float(s))
    theta_S = np.array(theta_S).reshape(-1, 1)

    theta_H = thetas.loc[0, 'theta_H']
    theta_H = theta_H.replace('[', '')
    theta_H = theta_H.replace(']', '')
    split = theta_H.split('\n')
    theta_H = []
    for s in split:
        theta_H.append(float(s))
    theta_H = np.array(theta_H).reshape(-1, 1)

    theta_R = thetas.loc[0, 'theta_R']
    theta_R = theta_R.replace('[', '')
    theta_R = theta_R.replace(']', '')
    split = theta_R.split('\n')
    theta_R = []
    for s in split:
        theta_R.append(float(s))
    theta_R = np.array(theta_R).reshape(-1, 1)

    return theta_G, theta_S, theta_H, theta_R


def check_args(args):
    if len(args) < 3:
        print("give 2 .csv file name as argument")
        exit()
    if args[1][-4:] != ".csv" or args[2][-4:] != ".csv":
        print("give .csv file name as argument")
        exit()
    if args[1][-8:] == "test.csv" and args[2] == "theta.csv":
        file_test = args[1]
        file_theta = args[2]
    elif args[2][-8:] == "test.csv" and args[1] == "theta.csv":
        file_test = args[2]
        file_theta = args[1]
    else:
        print(
            "give a train.csv file name as argument and other containing theta"
        )
        exit()
    return file_test, file_theta


if __name__ == "__main__":
    argv = sys.argv
    file_test, file_theta = check_args(argv)
    df = get_data(file_test)
    # keep only 8 schools subject columns and index
    # need to keep the index because of dropna
    df = df[['Index',
             'Herbology',
             'Ancient Runes',
             'Astronomy',
             'Defense Against the Dark Arts',
             'Charms',
             'Divination',
             'Potions',
             'History of Magic']]
    df = df.dropna()
    # copy for keep index after training
    copy_df = copy.deepcopy(df)

    # no need to keep the index to train or predict
    df = df.drop(columns=['Index'])
    describe_df = describe(df, ['Herbology',
                                'Ancient Runes',
                                'Astronomy',
                                'Defense Against the Dark Arts',
                                'Charms',
                                'Divination',
                                'Potions',
                                'History of Magic'])
    for col in df:
        df[[col]] = (
            df[[col]] - describe_df[col]['min']) / (
            describe_df[col]['max'] - describe_df[col]['min']
        )

    x = df.to_numpy()

    theta_G, theta_S, theta_H, theta_R = from_theta_csv_to_np(file_theta)

    # get predictions
    y_hat_G = logreg_predict(x, theta_G)
    y_hat_S = logreg_predict(x, theta_S)
    y_hat_H = logreg_predict(x, theta_H)
    y_hat_R = logreg_predict(x, theta_R)

    # from prediction to Hogwarts House
    result = from_y_hats_to_house(y_hat_G, y_hat_S, y_hat_R, y_hat_H, copy_df)

    print(result[['Index', 'Hogwarts House']])
    result = result[['Index', 'Hogwarts House']]
    print(result)
    result.to_csv('houses.csv', index=False)
