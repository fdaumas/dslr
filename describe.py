import sys

from getData import get_data
from listNumerical import list_numerical
from statsTools import mean, count, std, min, max, quantile


def describe(df, list_col):
    print(f"{7* ' '}", end="| ")
    for i in range(list_col.shape[0]):
        print(f"{list_col[i][-13:]:>13s}", end=' | ')
    print(f"\n{'count':>7s}", end="| ")
    for i in range(list_col.shape[0]):
        print(f"{str(count(df, list_col[i]))[:13]:>13s}", end=' | ')
    print(f"\n{'mean':>7s}", end="| ")
    for i in range(list_col.shape[0]):
        print(f"{str(mean(df, list_col[i]))[:13]:>13s}", end=' | ')
    print(f"\n{'std':>7s}", end="| ")
    for i in range(list_col.shape[0]):
        print(f"{str(std(df, list_col[i]))[:13]:>13s}", end=' | ')
    print(f"\n{'min':>7s}", end="| ")
    for i in range(list_col.shape[0]):
        print(f"{str(min(df, list_col[i]))[:13]:>13s}", end=' | ')
    print(f"\n{'25%':>7s}", end="| ")
    for i in range(list_col.shape[0]):
        print(f"{str(quantile(df, list_col[i], 0.25))[:13]:>13s}", end=' | ')
    print(f"\n{'50%':>7s}", end="| ")
    for i in range(list_col.shape[0]):
        print(f"{str(quantile(df, list_col[i], 0.50))[:13]:>13s}", end=' | ')
    print(f"\n{'75%':>7s}", end="| ")
    for i in range(list_col.shape[0]):
        print(f"{str(quantile(df, list_col[i], 0.75))[:13]:>13s}", end=' | ')
    print(f"\n{'max':>7s}", end="| ")
    for i in range(list_col.shape[0]):
        print(f"{str(max(df, list_col[i]))[:13]:>13s}", end=" | ")
    print('')


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        print("give .csv file name as argument")
        exit()
    file_name = argv[1]
    if file_name[-4:] != ".csv":
        print("give .csv file name as argument")
        exit()
    df = get_data(file_name)
    list_col = list_numerical(df)
    describe(df, list_col)
