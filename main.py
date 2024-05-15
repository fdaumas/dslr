import sys
import copy

from getData import get_data
from listNumerical import list_numerical
from describe import describe
from histogram import histogram
from scatter_plot import scatter_plot
from pair_plot import pair_plot


if __name__ == '__main__':
    # text color
    reset = '\033[0m'
    bold = '\033[01m'
    green = '\033[32m'
    red = '\033[31m'
    yellow = '\033[33m'
    blue = '\033[34m'


    argv = sys.argv
    if len(argv) < 2:
        print(red + "give .csv file name as argument" + reset)
        exit()
    file_name = argv[1]
    if file_name[-4:] != ".csv":
        print(red + "give .csv file name as argument" + reset)
        exit()
    df = get_data(file_name)
    org_list_col = list_numerical(df)


    list_ex = [
        "Data analisys",
        "Data visualisation",
        "Machine learning",
        "Quit"
    ]
    list_visu = ["Histogram", "Scatter plot", "Pair plot", "Back", "Quit"]

    while True:
        ex = 'a'
        while not ex.isdigit() or int(ex) < 1 or int(ex) > 4:
            for choice in list_ex:
                print(f"{green}{list_ex.index(choice)+1}. {choice}{reset}")
            ex = input(bold + "Do you want to see: " + reset)
        ex = int(ex)
        match ex:
            case 1:  # Data Analysis
                new_file_name = file_name[:-4]
                split = new_file_name.split('/')
                print(describe(df, org_list_col, split[-1] + "_describe.csv"))
                continue

            case 2:  # Data Visualisation
                visu = 'a'
                while not visu.isdigit() or int(visu) < 1 or int(visu) > 5:
                    for choice in list_visu:
                        print(f"{yellow}{list_visu.index(choice)+1}. {choice}{reset}")
                    visu = input(bold + "Do you want to see: " + reset)
                visu = int(visu)
                match visu:
                    case 1:  # Histogram
                        histogram(df)
                        continue

                    case 2:  # scatter plot
                        col1 = 'a'
                        print(blue + "Choose the 1st column for scatter plot:" + reset)
                        for i in range(len(org_list_col)):
                            print(f"{blue}{i+1}. {org_list_col[i]}{reset}")
                        while (
                            not col1.isdigit()
                            or int(col1) < 1 or int(col1) > len(org_list_col)
                        ):
                            col1 = input(bold + "Enter the column number: " + reset)
                        col1 = int(col1)
                        list_col = copy.deepcopy(org_list_col)
                        list_col = list_col.drop(org_list_col[col1-1])
                        print(blue + "Choose the 2nd column for scatter plot:" + reset)
                        for i in range(len(list_col)):
                            print(f"{blue}{i+1}. {list_col[i]}{reset}")
                        col2 = 'a'
                        while (
                            not col2.isdigit()
                            or int(col2) < 1 or int(col2) > len(list_col)
                        ):
                            col2 = input(bold + "Enter the column number: " + reset)
                        scatter_plot(
                            df,
                            org_list_col[col1-1],
                            list_col[int(col2)-1]
                        )

                    case 3:  # Pair plot
                        pair_plot(df)
                        continue

                    case 4:  # Back
                        ex = 'a'
                        continue

                    case 5:  # Quit
                        exit()

                continue

            case 3:  # Machine Learning
                # TODO
                continue

            case 4:  # Quit
                exit()
