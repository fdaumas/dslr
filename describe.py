import sys
import pandas as pd

from getData import get_data
from listNumerical import list_numerical
from statsTools import mean, count, std, min, max, quantile
from statsTools import range, count_over_mean


# def describe(df, list_col):
#         print(f"{7* ' '}", end="| ")
#         for i in range(list_col.shape[0]):
#             my_count = count(df, list_col[i])
#             if my_count != 0:
#                 print(f"{list_col[i][-13:]:>13s}", end=' | ')
#                 # '[-13:]' pour limiter a 13 char
#                 # ':>13s' pour aligner a droite
#                 # "end=' | '" pour echanger '\n' avec ' | '
#         print(f"\n{'count':>7s}", end="| ")
#         for i in range(list_col.shape[0]):
#             my_count = count(df, list_col[i])
#             if my_count != 0:
#                 print(f"{str(my_count)[:13]:>13s}", end=' | ')
#         print(f"\n{'mean':>7s}", end="| ")
#         for i in range(list_col.shape[0]):
#             my_count = count(df, list_col[i])
#             if my_count != 0:
#                 print(f"{str(mean(df, list_col[i]))[:13]:>13s}", end=' | ')
#         print(f"\n{'std':>7s}", end="| ")
#         for i in range(list_col.shape[0]):
#             my_count = count(df, list_col[i])
#             if my_count != 0:
#                 print(f"{str(std(df, list_col[i]))[:13]:>13s}", end=' | ')
#         print(f"\n{'min':>7s}", end="| ")
#         for i in range(list_col.shape[0]):
#             my_count = count(df, list_col[i])
#             if my_count != 0:
#                 print(f"{str(min(df, list_col[i]))[:13]:>13s}", end=' | ')
#         print(f"\n{'25%':>7s}", end="| ")
#         for i in range(list_col.shape[0]):
#             my_count = count(df, list_col[i])
#             if my_count != 0:
#                 print(f"{str(quantile(df, list_col[i], 0.25))[:13]:>13s}", end=' | ')
#         print(f"\n{'50%':>7s}", end="| ")
#         for i in range(list_col.shape[0]):
#             my_count = count(df, list_col[i])
#             if my_count != 0:
#                 print(f"{str(quantile(df, list_col[i], 0.50))[:13]:>13s}", end=' | ')
#         print(f"\n{'75%':>7s}", end="| ")
#         for i in range(list_col.shape[0]):
#             my_count = count(df, list_col[i])
#             if my_count != 0:
#                 print(f"{str(quantile(df, list_col[i], 0.75))[:13]:>13s}", end=' | ')
#         print(f"\n{'max':>7s}", end="| ")
#         for i in range(list_col.shape[0]):
#             my_count = count(df, list_col[i])
#             if my_count != 0:
#                 print(f"{str(max(df, list_col[i]))[:13]:>13s}", end=" | ")
#         print('')


def describe(df, list_col, filename="describe.csv"):
    list_stats = [
        'count',
        'mean',
        'std',
        'min',
        '25%',
        '50%',
        '75%',
        'max',
        'range',
        'count over mean'
    ]
    new_df = pd.DataFrame(index=list_stats, columns=list_col)
    for col in list_col:
        # new_df[col]['count'] = count(df, col)
        new_df.loc['count', col] = count(df, col)
        if new_df.loc['count', col] == 0:
            new_df.drop(col, axis=1, inplace=True)
            continue
        # new_df[col]['mean'] = mean(df, col)
        new_df.loc['mean', col] = mean(df, col)
        # new_df[col]['std'] = std(df, col)
        new_df.loc['std', col] = std(df, col)
        # new_df[col]['min'] = min(df, col)
        new_df.loc['min', col] = min(df, col)
        # new_df[col]['25%'] = quantile(df, col, 0.25)
        new_df.loc['25%', col] = quantile(df, col, 0.25)
        # new_df[col]['50%'] = quantile(df, col, 0.50)
        new_df.loc['50%', col] = quantile(df, col, 0.50)
        # new_df[col]['75%'] = quantile(df, col, 0.75)
        new_df.loc['75%', col] = quantile(df, col, 0.75)
        # new_df[col]['max'] = max(df, col)
        new_df.loc['max', col] = max(df, col)
        new_df.loc['range', col] = range(df, col)
        new_df.loc['count over mean', col] = count_over_mean(df, col)

    # write in a csv
    new_df.to_csv(filename)

    return new_df


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

    # describe(df, list_col)

    file_name = file_name[:-4]
    split = file_name.split('/')

    print(describe(df, list_col, split[-1] + "_describe.csv"))
