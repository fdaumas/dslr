import sys
import seaborn as sns
import matplotlib.pyplot as plt

from getData import get_data


def pair_plot(df):
    house = df['Hogwarts House'].unique()
    hue_order = []
    if 'Ravenclaw' in house:
        hue_order.append('Ravenclaw')
    if 'Hufflepuff' in house:
        hue_order.append('Hufflepuff')
    if 'Slytherin' in house:
        hue_order.append('Slytherin')
    if 'Gryffindor' in house:
        hue_order.append('Gryffindor')
    sns.pairplot(
        df,
        diag_kind='hist',
        hue='Hogwarts House',
        hue_order=hue_order,
        corner=True,
    )
    plt.show()


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
    pair_plot(df)
