import pandas as pd


def get_data(file_name):
    try:
        csv = pd.read_csv(file_name)
    except:
        print("the file can't be open")
        exit()
    return csv


if __name__ == '__main__':
    df = get_data("./datasets/dataset_train.csv")
    print(df.head())
