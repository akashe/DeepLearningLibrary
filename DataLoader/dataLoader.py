import os
import random
import torch
from .csvLoader import csvLoader
from .xlsLoader import xlsLoader
import sys


def dataLoader(file_path, file_name, split_ratio=0.8, remove_first_column=False):
    if file_name.endswith('.csv'):
        X, Y = csvLoader(file_path, file_name, remove_first_column=False)
    elif file_name.endswith('.xls'):
        X, Y = xlsLoader(file_path, file_name, remove_first_column=False)
    else:
        print("File format not supported yet")
        sys.exit(1)

    # randomizing the inputs
    # TODO: make this randomizer a seperate function itself
    a = list(range(len(X)))
    random.shuffle(a)
    split_index = int(len(a) * split_ratio)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i, index in enumerate(a):
        if i <= split_index:
            train_x.append(map(float, X[index]))
            train_y.append(float(Y[index]))
        else:
            test_x.append(map(float, X[index]))
            test_y.append(float(Y[index]))

    return torch.FloatTensor(train_x), torch.FloatTensor(train_y), torch.FloatTensor(test_x), torch.FloatTensor(test_y)
