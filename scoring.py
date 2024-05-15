import numpy as np

from getData import get_data
from logreg_predict import logreg_predict, from_theta_csv_to_np
from logreg_predict import from_y_hats_to_house


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
        if y_house[i] == y_train[i]:
            correct += 1
    return correct / m


if __name__ == "__main__":
    df_train = get_data("datasets/dataset_train.csv")
    df_train = df_train[["Hogwarts House", "Herbology", "Ancient Runes"]]
    theta_GorH, theta_GorS = from_theta_csv_to_np("theta.csv")
    result = from_y_hats_to_house(theta_GorH, theta_GorS)
    y_train = df_train["Hogwarts House"].to_numpy()
    y_house = result["Hogwarts House"].to_numpy()
    print(scoring(y_house, y_train))
