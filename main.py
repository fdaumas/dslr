import sys

from getData import get_data
from listNumerical import list_numerical


if __name__ == '__main__':
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
    print(df.head())
