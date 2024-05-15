from getData import get_data


if __name__ == "__main__":
    df_train = get_data("datasets/dataset_train.csv")
    df_test = get_data("datasets/dataset_test.csv")
    print(df_train)
    print(df_test)