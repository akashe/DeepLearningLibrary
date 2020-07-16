from pandas import DataFrame, read_csv
import pandas as pd
import os


# TODO: add support for huge files, non ascii characters and multiple sheets

def xlsLoader(file_path, file_name, remove_first_column):
    df = pd.read_excel(os.path.join(file_path, file_name))
    if remove_first_column:
        XY = df.values[:,1:]
    else:
        XY = df.values
    # Assumption: Last row is target value
    X = []
    Y = []
    for row in XY:
        X.append(row[:-1])
        Y.append(row[-1])
    return X, Y
