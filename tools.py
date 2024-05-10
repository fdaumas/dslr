from getData import get_data
from listNumerical import list_numerical


import pandas as pd
import math


def find_NA(df, col):
    """
        argument: 
            df : dataframe
            col : column of df

        return: 
            number of missing values in the column col
    """
    for i in range(len(df[col])):
        if math.isnan(df[col][i]):
            print(f"NA found at {col} - {i}")
        break


if __name__ == "__main__":
    df = get_data("datasets/dataset_train.csv")
    collist = list_numerical(df)
    for col in collist:
        find_NA(df, col)
    # for col in df.columns:
    #     find_NA(df, col)
    print(df.head())