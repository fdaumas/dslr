import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from getData import get_data
from listNumerical import list_numerical
from describe import describe
from statsTools import std
import sys


def histogram(df):
    house = np.unique(df[['Hogwarts House']])
    if len(house) < 1:
        print('this csv not include a single Hogwarts House')
        exit()

    num_df = list_numerical(df)
    desc = describe(df, num_df)

    new_df = copy.deepcopy(df)

    for col in num_df:
        new_df[[col]] = (df[[col]] - desc[col]['min']) / (desc[col]['max'] - desc[col]['min'])

    df_house = new_df.groupby(['Hogwarts House'])
    new_df_house_sort = pd.DataFrame(index=house, columns=num_df)
    for col in num_df:
        for h in house:
            x = df_house.get_group((h,))
            new_df_house_sort.loc[h, col] = std(x, col)
    new_df_house_sort = new_df_house_sort.dropna()
    print(new_df_house_sort)

    fig, axs = plt.subplots(1, len(house), tight_layout=True)

    for index in new_df_house_sort:
        print(index)
        # N, bins, patches = axs.hist[index](new_df_house_sort.iloc[[index]], bins=len(num_df))
    # hey = new_df_house_sort.hist(bins=len(num_df))

    new_df_house_sort = new_df_house_sort.transpose()
    print(new_df_house_sort)
    for i in range(len(house)):
        axs[i].hist(new_df_house_sort[house[i]], bins=len(num_df))

    # for i in range(len(house)):
    #     cp = copy.deepcopy(df_house.get_group((house[i],)))
    #     dist = []
    #     for col in num_df:
    #         res_std = std(cp, col)
    #         dist.append(res_std)
    #     fracs = N / N.max()
    #     norm = colors.Normalize(fracs.min(), fracs.max())
    #     for thisfrac, thispatch in zip(fracs, patches):
    #         color = plt.cm.viridis(norm(thisfrac))
    #         thispatch.set_facecolor(color)
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
    # print("Choose the 1st column for scatter plot:")
    # for i in range(len(org_list_col)):
    #     print(f"{i+1}. {org_list_col[i]}")
    # col1 = 'a'
    # while not col1.isdigit():
    #     col1 = input("Enter the column number: ")
    # col1 = int(col1)
    # list_col = copy.deepcopy(org_list_col)
    # list_col = list_col.drop(org_list_col[col1-1])
    # print("Choose the 2nd column for scatter plot:")
    # for i in range(len(list_col)):
    #     print(f"{i+1}. {list_col[i]}")
    # col2 = 'a'
    # while not col2.isdigit():
    #     col2 = input("Enter the column number: ")
    # scatter_plot(df, org_list_col[col1-1], list_col[int(col2)-1])

    histogram(df)
