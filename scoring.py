import numpy as np
import pandas as pd
import copy

from color import red, yellow, green, blue, reset
from getData import get_data
from describe import describe
from logreg_predict import logreg_predict, from_theta_csv_to_np
from logreg_predict import from_y_hats_to_house


def print_house(house, end):
    if house == 'Gryffindor':
        print(red + house, end=end)
    if house == 'Slytherin':
        print(green + house, end=end)
    if house == 'Hufflepuff':
        print(yellow + house, end=end)
    if house == 'RavenClaw':
        print(blue + house, end=end)


def scoring(y_house, y_train):
    """Check the accuracy of the prediction

    Args:
        x_house (numpy.ndarray): the prediction
        x_train (numpy.ndarray): the expected value
    Returns:
        float: the accuracy of the prediction
    """
    if not isinstance(y_house, np.ndarray) or not isinstance(y_train, np.ndarray):
        return None
    if y_house.size == 0 or y_train.size == 0:
        return None
    if y_house.shape != y_train.shape:
        return None

    m = y_house.shape[0]
    correct = 0
    for i in range(m):
        print_house(y_house[i], ' | ')
        print_house(y_train[i], '\n')
        print(reset + f"index = {i}")
        if y_house[i] == y_train[i]:
            correct += 1
    print(correct)
    return correct / m


def predict_train():
    df = get_data("datasets/dataset_train.csv")
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

    theta_GorH, theta_GorS = from_theta_csv_to_np('theta.csv')

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

    result = result[['Index', 'Hogwarts House']]
    print(result)
    return result


if __name__ == "__main__":
    df_train = get_data("datasets/dataset_train.csv")
    df_train.dropna()
    df_train = df_train[["Hogwarts House", "Index"]]

    df_house = predict_train()
    df_house.dropna()
    print(df_house.head(40))
    df_house = df_train[["Hogwarts House", "Index"]]

    y_train = df_train["Hogwarts House"].to_numpy()
    y_house = df_house["Hogwarts House"].to_numpy()
    print(reset + '')
    print(scoring(y_house, y_train))
