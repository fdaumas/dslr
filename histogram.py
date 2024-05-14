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
    color = {'Gryffindor': '#ae0001', 'Hufflepuff': '#ecb939', 'Ravenclaw': '#1f82cc', 'Slytherin': '#096A09'}
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

    new_df_house_sort = new_df_house_sort.transpose()
    for i in range(len(house)):
        axs[i].hist(new_df_house_sort[house[i]], bins=len(num_df), color=color[house[i]])
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
    histogram(df)
