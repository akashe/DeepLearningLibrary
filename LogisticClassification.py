import argparse
from abc import ABC

from CostFunctions import CrossEntropy
from DataLoader import get_data,Dataset
from Modules import Linear
from Modules import Relu
from Models import Model
from Optimizers import Optimizer,SGDOptimizerForModules
import torch
from torch.utils.data import DataLoader


'''
There should be easier way to access these classes
Create a model class
'''


class LogisticClassification(Model, ABC):
    def __init__(self, layer1dim, layer2dim, layer3dim,optim):
        super().__init__(optim)
        self.layer1 = Linear(layer1dim)
        self.relu1 = Relu()
        # Ohhh shit I see problems defining it this way
        self.layer2 = Linear(layer2dim)
        self.relu2 = Relu()
        self.layer3 = Linear(layer3dim)
        self.loss = CrossEntropy()
        self.loss_ = None

    def forward(self, inputs, targets):
        a = self.layer1(inputs)
        a = self.relu1(a)
        a = self.layer2(a)
        a = self.relu2(a)
        a = self.layer3(a)

        self.loss_ = self.loss(a, targets)
        acc_ = (torch.argmax(a.o, dim=1) == targets.long()).float().mean()
        print(self.loss_.o)
        print("Accuracy ="+str(acc_))
        return self.loss_.o

    def backward(self):
        '''
        TODO: update this later..too inefficient
        Debugging module.backward()
        '''
        self.loss_.backward()


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

    train_x, train_y, test_x, test_y = get_data(args.filepath, args.filename)
    train_ds = Dataset(train_x,train_y) # a dataset just gives a __getitem__
    test_ds = Dataset(test_x,test_y)

    train_dl = DataLoader(dataset=train_ds,batch_size=int(args.batch_size))
    # a dataloader does 2 things: shuffles the data and convert values to a tensor
    test_dl = DataLoader(dataset=test_ds,batch_size=2*int(args.batch_size))
    torch.autograd.set_detect_anomaly(True)
    no_of_classes = 1 + int(test_y.max().item())
    # optim = Optimizer(model.trainable_params,float(args.learning_rate))
    optim = SGDOptimizerForModules(float(args.learning_rate))
    model = LogisticClassification([len(train_x[0]), 100], [100, 50], [50, no_of_classes], optim)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    for i in range(epochs):
        for j,(x,y) in enumerate(train_dl):
            print(" Train error for epoch " + str(i) + " and batch " + str(j + 1) + " : ")
            model.forward(x,y)
            model.backward()
            # optim.step()
            # optim.zero_grad()

        for j,(x,y) in enumerate(test_dl):
            print(" Test error for epoch " + str(i) + " and batch " + str(j + 1) + " : ")
            with torch.no_grad():
                model.forward(x,y)


if __name__ == "__main__":
    main()
