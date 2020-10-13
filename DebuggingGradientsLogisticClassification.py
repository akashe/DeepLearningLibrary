from Models import Model
from abc import ABC
from Modules import LinearWithHooks, Relu, BatchNorm
from CostFunctions import CrossEntropy
import argparse
from DataLoader import get_data, Dataset
from torch.utils.data import DataLoader
from Optimizers import Optimizer
from Callbacks import LossAndAccuracyCallback,LrScheduler,cosine_schedule
from RunUtils import Runner
from matplotlib import pyplot as plt
import time

"""
Idea: plot gradients for W's and b's in linear layer..check their mean and variance and see
if becoming zeros
Method: Simple logistic classification on MNIST. 
Experiments: check variations in gradients with BatchNorm, kaiming_initilization
Note: 
TODO: Ideally there should be something in Model class that can scoop up the stats and later plot them with Callbacks
"""


# TODO: make a sequential model class to avoid writing it again and again
class LogisticClassification(Model, ABC):
    def __init__(self, layer1dims, layer2dims, layer3dims, batch_norm=False):
        super(LogisticClassification, self).__init__()
        self.layer1 = LinearWithHooks(layer1dims)
        self.relu1 = Relu()
        self.layer2 = LinearWithHooks(layer2dims)
        self.relu2 = Relu()
        self.layer3 = LinearWithHooks(layer3dims)
        self.loss = CrossEntropy()
        self.loss_ = None
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = BatchNorm()
            self.bn2 = BatchNorm()

    def forward(self, inputs, targets):
        input_ = self.layer1(inputs)
        input_ = self.relu1(input_)
        if self.batch_norm:
            input_ = self.bn1(input_)
        input_ = self.layer2(input_)
        input_ = self.relu2(input_)
        if self.batch_norm:
            input_ = self.bn2(input_)
        input_ = self.layer3(input_)
        self.loss_ = self.loss(input_, targets)
        return self.loss_.o, input_.o

    def backward(self):
        # I know I know...this shud be updated
        self.loss_.o.backward()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="directory containing the file")
    parser.add_argument("--filename", help="file containing data")
    parser.add_argument("--epochs", type=int, help="total number of epochs")
    parser.add_argument("--batch_size", type=int, help="size of a batch")
    parser.add_argument("--learning_rate", type=float, help="learning rate if using SGD")
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = get_data(args.filepath, args.filename)
    train_ds = Dataset(train_x, train_y)
    test_ds = Dataset(test_x, test_y)

    train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size)
    test_dl = DataLoader(dataset=test_ds, batch_size=2 * args.batch_size)

    no_of_classes = 1 + (train_y.max().item())
    model = LogisticClassification([len(train_x[0]), 100], [100, 50], [50, no_of_classes])
    optim = Optimizer(model.trainable_params, args.learning_rate)
    callbacks = [LossAndAccuracyCallback(),LrScheduler(cosine_schedule(args.learning_rate,0.01))]
    runner = Runner(model,optim,train_dl,test_dl,callbacks)

    runner.fit(args.epochs)
    # This is the problem with individual param hooks, will have to plot them seperately
    # Its better to get gradients for entire layers

    #TODO: add better plots

    # Mean of gradients of W
    plt.plot(model.layer1.grads_mean[0], label="Layer_1_W")
    plt.plot(model.layer2.grads_mean[0], label="Layer_2_W")
    plt.plot(model.layer3.grads_mean[0], label="Layer_3_W")
    plt.legend(loc="upper right")
    plt.title("Means of Weights")
    plt.show()
    plt.close()

    # Variances of gradients of W
    plt.plot(model.layer1.grads_variance[0],label="Layer_1_W")
    plt.plot(model.layer2.grads_variance[0],label="Layer_2_W")
    plt.plot(model.layer3.grads_variance[0],label="Layer_3_W")
    plt.legend(loc="upper right")
    plt.title("Variances of Weights")
    plt.show()


if __name__ == "__main__":
    main()
