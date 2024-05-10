import math
import numpy as np
import copy


from getData import get_data
from listNumerical import list_numerical


def count(df, col):
    """
        argument:
            df : dataframe
            col : column of df

        return:
            count of the column col
    """
    count = 0
    for i in range(len(df[col])):
        if not math.isnan(df[col][i]):
            count += 1
    return count


def mean(df, col):
    """
        argument:
            df : dataframe
            col : column of df

        return:
            mean of the column col
    """
    data = df[col]
    sum = 0
    for i in range(len(data)):
        if not math.isnan(data[i]):
            sum += data[i]
    sum /= count(df, col)
    return sum


def std(df, col):
    """
        argument:
            df : dataframe
            col : column of df

        return:
            standard deviation of the column col
    """
    my_mean = mean(df, col)
    res = math.sqrt(sum([(x - my_mean) ** 2 for x in df[col] if not math.isnan(x)]) / count(df, col))
    return res


def min(df, col):
    """
        argument:
            df : dataframe
            col : column of df

        return:
            minimum value of the column col
    """
    data = df[col]
    min = data[0]
    for i in range(len(data)):
        if data[i] < min:
            min = data[i]
    return min


def quantile(df, col, q):
    """
        argument:
            df : dataframe
            col : column of df
            q : quantile number (0.25, 0.5, 0.75)

        return:
            q quantile of the column col
    """
    data = copy.deepcopy(df[col])
    data = data.dropna()
    data = np.array(df[col])
    data.sort()
    position = count(df, col) * q
    if position == int(position):
        position = int(position) + 1
    else:
        position = int(position)
    return data[position]


def max(df, col):
    """
        argument:
            df : dataframe
            col : column of df

        return:
            maximum value of the column col
    """
    data = df[col]
    max = data[0]
    for i in range(len(data)):
        if data[i] > max:
            max = data[i]
    return max


if __name__ == "__main__":
    df = get_data("datasets/dataset_train.csv")
    collist = list_numerical(df)
    for col in collist:
        print(f"{col} :")
        print(f"count : {count(df, col)}")
        # print(f"true count : {df[col].count()}")
        print(f"mean : {mean(df, col)}")
        # print(f"true mean : {df[col].mean()}")
        print(f"std : {std(df, col)}")
        # print(f"true std : {df[col].std()}")
        print(f"min : {min(df, col)}")
        # print(f"true min : {df[col].min()}")
        print(f"25% : {quantile(df, col, 0.25)}")
        # print(f"true 25% : {df[col].quantile(0.25)}")
        print(f"50% : {quantile(df, col, 0.5)}")
        # print(f"true 50% : {df[col].quantile(0.5)}")
        print(f"75% : {quantile(df, col, 0.75)}")
        # print(f"true 75% : {df[col].quantile(0.75)}")
        print(f"max : {max(df, col)}")
        # print(f"true max : {df[col].max()}")
        print("\n")

    # for verify par hand if quantile is correct
    # col = 'Potions'
    # data = copy.deepcopy(df[col])
    # data = data.dropna()
    # data = np.array(df[col])
    # data.sort()
    # print(data)
    # for i in range(0, 10):
    #     print(f"{i}\t{data[i]}")
    # print("...")
    # print(count(df, col)*0.25)
    # for i in range(387, 397):
    #     print(f"{i}\t{data[i]}")
    # print("...")
    # print(count(df, col)*0.5)
    # for i in range(780, 790):
    #     print(f"{i}\t{data[i]}")
    # print("...")
    # print(count(df, col)*0.75)
    # for i in range(1172, 1182):
    #     print(f"{i}\t{data[i]}")
    # print("...")
    # for i in range(1560, 1570):
    #     print(f"{i}\t{data[i]}")
