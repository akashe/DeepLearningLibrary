from Models.LinearModel import LinearModel
from DataLoader.dataLoader import dataLoader
import argparse

'''
    The idea is to simply:
    a) create a linear approximation of the data; equation of the form y = XM
    b) first use Mean Squared Error; later go to Maximum likelihood
    c) Calculate derivatives
    d) train the model with gradient descent or with normal form

'''


def main():
    # TODO: shift to a config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="directory containing data")
    parser.add_argument("--filename", help="file containing data")
    parser.add_argument("--remove_first_column", help="Remove first column of data")
    parser.add_argument("--epochs", help="total number of epochs")
    parser.add_argument("--batch_size", help="size of a batch")
    parser.add_argument("--update_rule", help="Matrix or SGD for normal form update or stochastic gradient descent")
    parser.add_argument("--learning_rate", help="learning rate if using SGD")
    parser.add_argument("--loss_type", help="MSE or ML for mean squared error or maximum likelihood")
    parser.add_argument("--regularization", help="L1 or L2 regularization")
    parser.add_argument("--regularization_constant", help="regularization constant")
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = dataLoader(args.filepath, args.filename, split_ratio=0.9,
                                                  remove_first_column=args.remove_first_column)

    # the issue is in the split..do I do the entire thing in one Go and check test error or
    #     I make batches of it .. I think I will do it in batches of 50

    linear_model = LinearModel(input_dim=len(train_x[0]), batch_size=args.batch_size, loss_type=args.loss_type,
                               update_rule=args.update_rule,
                               learning_rate=(float(args.learning_rate) if args.update_rule == "SGD" else 0.01),
                               regularization=args.regularization)

    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    for i in range(epochs):
        for j in range(int(len(train_x) / batch_size)):
            start_index = batch_size * j
            end_index = batch_size * (j + 1)
            print(" Train error for epoch " + str(i) + " and batch " + str(j + 1) + " : ")
            linear_model.run(train_x[start_index:end_index], train_y[start_index:end_index])

            # checking test error
            print(" Test error for epoch " + str(i) + " and batch " + str(j + 1) + " : ")
            linear_model.run(test_x, test_y, update_parameter=False)

    print("Akash")


if __name__ == "__main__":
    main()
