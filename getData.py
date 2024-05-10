import pandas as pd


def get_data(file_name):
    csv = pd.read_csv(file_name)
    return csv


if __name__ == '__main__':
    df = get_data("./datasets/dataset_train.csv")
    print(df.head())
