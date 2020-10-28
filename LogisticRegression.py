import argparse

from CostFunctions.MeanSquarredError import MSEloss
from DataLoader.dataLoader import dataLoader
from Modules.Linear import Linear
from Modules.Relu import Relu
from Models import Model
from Optimizers import SGDOptimizerForModules


class LogisticRegression(Model):
    def __init__(self, layer1dim, layer2dim, batch_size, optim):
        super(LogisticRegression, self).__init__(optim)
        self.layer1 = Linear(layer1dim)
        self.relu1 = Relu()
        self.layer2 = Linear(layer2dim)
        self.relu2 = Relu()
        self.loss = MSEloss(batch_size)
        self.loss_ = None

    def forward(self, inputs, targets):
        a = self.layer1(inputs)
        b = self.relu1(a)
        c = self.layer2(b)
        d = self.relu2(c)
        self.loss_ = self.loss(d, targets)
        print(self.loss_.o)

    def backward(self):
        self.loss_.backward()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="directory containing data")
    parser.add_argument("--filename", help="file containing data")
    parser.add_argument("--remove_first_column",type=bool, help="Remove first column of data")
    parser.add_argument("--epochs",type=int, help="total number of epochs")
    parser.add_argument("--batch_size",type= int, help="size of a batch")
    parser.add_argument("--update_rule", help="Matrix or SGD for normal form update or stochastic gradient descent")
    parser.add_argument("--learning_rate",type=float, help="learning rate if using SGD")
    parser.add_argument("--loss_type", help="MSE or ML for mean squared error or maximum likelihood")
    parser.add_argument("--regularization", help="L1 or L2 regularization")
    parser.add_argument("--regularization_constant", help="regularization constant")
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = dataLoader(args.filepath, args.filename, split_ratio=0.9,
                                                  remove_first_column=args.remove_first_column)

    optim = SGDOptimizerForModules(args.learning_rate)
    model = LogisticRegression([len(train_x[0]), 10], [10, 1], args.batch_size, optim)
    epochs = args.epochs
    batch_size = args.batch_size
    for i in range(epochs):
        for j in range(int(len(train_x) / batch_size)):
            start_index = batch_size * j
            end_index = batch_size * (j + 1)
            print(" Train error for epoch " + str(i) + " and batch " + str(j + 1) + " : ")
            model.forward(train_x[start_index:end_index], train_y[start_index:end_index])
            model.backward()


if __name__ == "__main__":
    main()
