import os


def textLoader(file_path, file_name, remove_first_column):
    print(" Using \t as dekimeter..update code for other delimeters")
    print(" Reading in non-binary mode..update code for binary files")
    with open(os.path.join(file_path, file_name), "r") as f:
        X = []
        Y = []
        for line in f:
            line = line.strip().split("\t")
            a = 1 if remove_first_column else 0
            X.append(line[a:-1])
            Y.append(line[-1])
    return X, Y
