import argparse

from CostFunctions.MeanSquarredError import MSEloss
from DataLoader.dataLoader import dataLoader
from Modules.Linear import Linear
from Modules.Relu import Relu
from Modules.module import Module

'''
There should be easier way to access these classes
Create a model class
'''


class LogisticRegression():
    def __init__(self, layer1dim, layer2dim, batch_size, lr):
        self.layer1 = Linear(layer1dim)
        self.relu1 = Relu()
        # Ohhh shit I see problems defining it this way
        self.layer2 = Linear(layer2dim)
        self.relu2 = Relu()
        self.loss = MSEloss(batch_size)
        self.loss_ = None
        self.lr = lr

    def forward(self, inputs, targets):
        a = self.layer1(inputs)
        b = self.relu1(a)
        c = self.layer2(b)
        d = self.relu2(c)
        self.loss_ = self.loss(d, targets)
        print(self.loss_.o)

    def backward(self):
        '''
        TODO: update this later..too inefficient
        backward function of module class still buggy..using torch's backward in the mean
         time
        '''
        self.loss_.o.backward()
        for i in vars(self):
            if isinstance(self.__getattribute__(i), Module):
                self.__getattribute__(i).update_params_torch(self.lr)

        for i in vars(self):
            if isinstance(self.__getattribute__(i), Module):
                self.__getattribute__(i).set_grad_zero()


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
                                                  remove_first_column=(
                                                      True if args.remove_first_column == "True" else False))

    model = LogisticRegression([len(train_x[0]), 10], [10, 1], int(args.batch_size), float(args.learning_rate))
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    for i in range(epochs):
        for j in range(int(len(train_x) / batch_size)):
            start_index = batch_size * j
            end_index = batch_size * (j + 1)
            print(" Train error for epoch " + str(i) + " and batch " + str(j + 1) + " : ")
            model.forward(train_x[start_index:end_index], train_y[start_index:end_index])
            model.backward()


if __name__ == "__main__":
    main()
