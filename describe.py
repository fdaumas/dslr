import pandas as pd


def describe():
    csv = pd.read_csv("./datasets/dataset_train.csv")
    # for i in range(5):
    #     print(csv[i])
    print(csv.columns)
    print(csv[['Ancient Runes', 'Arithmancy']])


if __name__ == "__main__":
    describe()
