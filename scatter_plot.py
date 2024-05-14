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
    if len(argv) < 2:
        print("give .csv file name as argument")
        exit()
    file_name = argv[1]
    if file_name[-4:] != ".csv":
        print("give .csv file name as argument")
        exit()
    df = get_data(file_name)
    org_list_col = list_numerical(df)
    print("Choose the 1st column for scatter plot:")
    for i in range(len(org_list_col)):
        print(f"{i+1}. {org_list_col[i]}")
    col1 = 'a'
    while not col1.isdigit():
        col1 = input("Enter the column number: ")
    col1 = int(col1)
    list_col = copy.deepcopy(org_list_col)
    list_col = list_col.drop(org_list_col[col1-1])
    print("Choose the 2nd column for scatter plot:")
    for i in range(len(list_col)):
        print(f"{i+1}. {list_col[i]}")
    col2 = 'a'
    while not col2.isdigit():
        col2 = input("Enter the column number: ")
    scatter_plot(df, org_list_col[col1-1], list_col[int(col2)-1])
