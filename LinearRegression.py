from Models.LinearModel import LinearModel
from dataLoader.csvLoader import csvLoader
import argparse

'''
    The idea is to simply:
    a) create a linear approximation of the data; equation of the form y = XM
    b) first use Mean Squared Error; later go to Maximum likelihood
    c) Calculate derivatives
    d) train the model with gradient descent or with normal form

'''


def main():
    # Later shift to a config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="directory containing data")
    parser.add_argument("--filename", help="file containing data")
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = csvLoader(args.filepath, args.filename, split_ratio=0.9,remove_first_column=True)

    # the issue is in the split..do I do the entire thing in one Go and check test error or
    #     I make batches of it .. I think I will do it in batches of 50

    linear_model = LinearModel(input_dim = len(train_x[0]), batch_size = 50, loss_type="MSE", update_rule="Matrix")
    print("Akash")


if __name__ == "__main__":
    main()
