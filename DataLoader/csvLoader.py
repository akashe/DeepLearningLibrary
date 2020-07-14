import csv
import os

'''
    TODO: Add tokenizer for string values
    TODO: Add yield for bigger files
'''


def csvLoader(file_path, file_name, remove_first_column=False):
    # type: (object, object, object, object) -> object
    with open(os.path.join(file_path, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        X = []
        Y = []
        #Assumption: Last row is target value
        for row in csv_reader:
            a = 1 if remove_first_column else 0
            X.append(row[a:-1])
            Y.append(row[-1])

    return X, Y
