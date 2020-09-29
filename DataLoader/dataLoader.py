import os
import pickle
import random
import torch
from .csvLoader import csvLoader
from .xlsLoader import xlsLoader
from .textLoader import textLoader
from .gzLoader import gzLoader
import sys
import gzip


def dataLoader(file_path, file_name, split_ratio=0.8, remove_first_column=False):
    if file_name.endswith('.csv'):
        X, Y = csvLoader(file_path, file_name, remove_first_column)
    elif file_name.endswith('.xls'):
        X, Y = xlsLoader(file_path, file_name, remove_first_column)
    elif file_name.endswith('.txt'):
        X, Y = textLoader(file_path, file_name, remove_first_column)
    elif file_name.endswith('.gz'):
        X, Y = gzLoader(file_path, file_name, remove_first_column)
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
        if i < split_index:
            train_x.append(list(map(float, X[index])))
            train_y.append(float(Y[index]))
        else:
            test_x.append(list(map(float, X[index])))
            test_y.append(float(Y[index]))
            # TODO : in classification tasks with cross entropy we dont need to make float of train_y and test_y

    return torch.FloatTensor(train_x), torch.FloatTensor(train_y), torch.FloatTensor(test_x), torch.FloatTensor(test_y)


def get_data(file_path, file_name):
    """
    #TODO : is there a better way to yield data from diectly here in a shuffled way?
    """
    with gzip.open(os.path.join(file_path, file_name), 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

    return x_train, y_train, x_valid, y_valid


class Dataset():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        assert len(x) == len(y)

    def __len__(self): return len(self.x)

    def __getitem__(self, item):
        return self.x[item],self.y[item]

