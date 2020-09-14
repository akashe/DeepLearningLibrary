from pandas import DataFrame, read_csv
import pandas as pd
import os


# TODO: add support for huge files, non ascii characters and multiple sheets

def xlsLoader(file_path, file_name, remove_first_column):
    df = pd.read_excel(os.path.join(file_path, file_name))
    XY = df.values
    X = []
    Y = []
    a = 1 if remove_first_column else 0
    for row in XY:
        X.append(row[a:-1])
        Y.append(row[-1])
    return X, Y
