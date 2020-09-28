import gzip
import os
import pickle
import numpy as np


def gzLoader(file_path, file_name, remove_first_column):
    # TODO: currently only for MNIST update later
    with gzip.open(os.path.join(file_path, file_name), 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

        return np.concatenate((x_train,x_valid)), np.concatenate((y_train,y_valid))
        # TODO: this is not right. It increases execution time. Specially in case of MNIST I dont have to convert
        # everything to float
