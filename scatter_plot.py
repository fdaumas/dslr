import sys
import matplotlib.pyplot as plt
import copy

from getData import get_data
from listNumerical import list_numerical


def scatter_plot(df, col1, col2):
    """
        argument:
            df : dataframe
            col1 : column of df
            col2 : column of df

        return:
            scatter plot of col1 and col2
    """
    plt.scatter(df[col1], df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f"{col1} vs {col2}")
    plt.show()


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 4:
        print("give .csv file name as argument")
        exit()
    file_name = argv[1]
    if file_name[-4:] != ".csv":
        print("give .csv file name as argument")
        exit()
    df = get_data(file_name)
    org_list_col = list_numerical(df)
    
    scatter_plot(df, argv[2], argv[3])