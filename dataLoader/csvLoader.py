import csv
import os
import random
import torch

'''
    TODO: Add tokenizer for string values
    TODO: Add yield for bigger files
'''


def csvLoader(file_path, file_name, split_ratio=0.8, remove_first_column=False):
    # type: (object, object, object, object) -> object
    with open(os.path.join(file_path, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        X = []
        Y =[]
        for row in csv_reader:
            a = 1 if remove_first_column else 0
            X.append(row[a:-1])
            Y.append(row[-1])

    # randomizing the inputs
    # TODO: make this randomizer a seperate function itself
    a = list(range(len(X)))
    random.shuffle(a)
    split_index = int(len(a)*split_ratio)
    train_x =[]
    train_y =[]
    test_x =[]
    test_y =[]
    for i, index in enumerate(a):
        if i <= split_index:
            train_x.append(map(float,X[index]))
            train_y.append(float(Y[index]))
        else:
            test_x.append(map(float,X[index]))
            test_y.append(float(Y[index]))

    return torch.FloatTensor(train_x), torch.FloatTensor(train_y), torch.FloatTensor(test_x), torch.FloatTensor(test_y)