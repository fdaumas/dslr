import sys
import copy

from getData import get_data
from listNumerical import list_numerical
from describe import describe
from histogram import histogram
from scatter_plot import scatter_plot
from pair_plot import pair_plot
from color import red, green, yellow, blue, reset, bold


def check_args(argv):
    if len(argv) < 3:
        print(red + "give 2 .csv file name as argument" + reset)
        exit()
    for i in range(1, 3):
        if argv[i][-4:] != ".csv":
            print(red + "give .csv file name as argument" + reset)
            exit()
    if argv[1] == argv[2]:
        print(red + "give 2 different .csv file name as argument" + reset)
        exit()
    if argv[1][-9:] == "train.csv" and argv[2][-8:] == "test.csv":
        file_train = argv[1]
        file_test = argv[2]
    elif argv[2][-9:] == "train.csv" and argv[1][-8:] == "test.csv":
        file_train = argv[2]
        file_test = argv[1]
    else:
        print(
            red
            + "give a train.csv file name and a test.csv as arguments"
            + reset
        )
        exit()
    return file_train, file_test


def menu_lvl0():
    list_ex = [
        "Data analisys",
        "Data visualisation",
        "Machine learning",
        "Quit"
    ]

    ex = 'a'
    while not ex.isdigit() or int(ex) < 1 or int(ex) > 4:
        for choice in list_ex:
            print(green + f"{list_ex.index(choice)+1}. {choice}" + reset)
        ex = input(bold + "What do you want to see: " + reset)
    ex = int(ex)
    return ex


def data_analysis(df_train, org_list_col_train):
    new_file_name = file_train[:-4]
    split = new_file_name.split('/')
    print(describe(df_train, org_list_col_train, split[-1] + "_describe.csv"))


def menu_lvl1_data_visu():
    list_visu = ["Histogram", "Scatter plot", "Pair plot", "Back", "Quit"]

    visu = 'a'
    while not visu.isdigit() or int(visu) < 1 or int(visu) > 5:
        for choice in list_visu:
            print(yellow + f"{list_visu.index(choice)+1}. {choice}" + reset)
        visu = input(bold + "What do you want to see: " + reset)
    visu = int(visu)
    return visu


def menu_lvl2_scatter_plot(org_list_col_train):
    col1 = 'a'
    print(blue + "Choose the 1st column for scatter plot:" + reset)
    for i in range(len(org_list_col_train)):
        print(blue + f"{i+1}. {org_list_col_train[i]}" + reset)
    while (
        not col1.isdigit()
        or int(col1) < 1
        or int(col1) > len(org_list_col_train)
    ):
        col1 = input(bold + "Enter the column number: " + reset)
    col1 = int(col1)
    list_col = copy.deepcopy(org_list_col_train)
    list_col = list_col.drop(org_list_col_train[col1-1])
    print(blue + "Choose the 2nd column for scatter plot:" + reset)
    for i in range(len(list_col)):
        print(f"{blue}{i+1}. {list_col[i]}{reset}")
    col2 = 'a'
    while (not col2.isdigit() or int(col2) < 1 or int(col2) > len(list_col)):
        col2 = input(bold + "Enter the column number: " + reset)
    return org_list_col_train[col1-1], list_col[int(col2)-1]


if __name__ == '__main__':
    argv = sys.argv
    file_train, file_test = check_args(argv)
    
    df_train = get_data(file_train)
    org_list_col_train = list_numerical(df_train)

    while True:
        ex = menu_lvl0()
        match ex:
            case 1:  # Data Analysis
                data_analysis(df_train, org_list_col_train)
                continue
            case 2:  # Data Visualisation
                visu = menu_lvl1_data_visu()
                match visu:
                    case 1:  # Histogram
                        histogram(df_train)
                        continue

                    case 2:  # scatter plot
                        col1, col2 = menu_lvl2_scatter_plot(org_list_col_train)
                        scatter_plot(df_train, col1, col2)

                    case 3:  # Pair plot
                        pair_plot(df_train)
                        continue

                    case 4:  # Back
                        continue

                    case 5:  # Quit
                        exit()
                continue
            case 3:  # Machine Learning
                # TODO
                continue
            case 4:  # Quit
                exit()
