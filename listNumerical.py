import numpy as np


from getData import get_data


def list_numerical(df):
    """
        argument:
            df : dataframe

        return:
            list of position columns of df that contains only numerical data
    """
    columns_of_numerical = df.columns
    # print(columns_of_numerical)
    for col in df.columns:
        my_col = df[col]
        # print(type(my_col[0]))
        for i in range(df.shape[0]):
            if (
                not isinstance(my_col[i], np.float64)
                and not isinstance(my_col[i], np.int64)
            ):
                columns_of_numerical = columns_of_numerical.drop(col)
                break
    return columns_of_numerical


if __name__ == "__main__":
    df = get_data("datasets/dataset_train.csv")
    print(df.shape)
    col = list_numerical(df)
    print(col)
