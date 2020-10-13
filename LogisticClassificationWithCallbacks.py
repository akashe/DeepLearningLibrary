import argparse
from abc import ABC

from CostFunctions import CrossEntropy
from DataLoader import get_data,Dataset
from Modules import Linear,BatchNorm
from Modules import Relu
from Models import Model
from Optimizers import Optimizer
import torch
from torch.utils.data import DataLoader
from RunUtils import Runner
from Callbacks import LossAndAccuracyCallback,LrScheduler,LrRecorder,cosine_schedule

'''
Main idea: to implement callbacks, their structure and 2 callbacks: lr scheduler and satistics callback
Also: testing batchNorm here, ideally should check it when conv2d is ready.. 

Experiments and results:
1) kaiming initilization has a drastic effect on accuracy
2) batch_norm doesnt really affect accuracies in this model,it particularly lowers it.
3) Lr scheduling didnt have much impact
4) higher learning rate of 0.1 gives 97% within 10 epochs
'''


class LogisticClassification(Model, ABC):
    def __init__(self, layer1dim, layer2dim, batch_size):
        super().__init__()
        self.layer1 = Linear(layer1dim)
        self.relu1 = Relu()
        self.bn1 = BatchNorm()
        # Ohhh shit I see problems defining it this way
        self.layer2 = Linear(layer2dim)
        self.loss = CrossEntropy()
        self.loss_ = None

    def forward(self, inputs, targets):
        a = self.layer1(inputs)
        b = self.bn1(a)
        b_ = self.relu1(b)
        d = self.layer2(b_)
        self.loss_ = self.loss(d, targets)
        return self.loss_.o, d.o

    def backward(self):
        '''
        TODO: update this later..too inefficient
        backward function of module class still buggy..using torch's backward in the mean
         time
        '''
        self.loss_.o.backward()


def main():
    # TODO: shift to a config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="directory containing data")
    parser.add_argument("--filename", help="file containing data")
    parser.add_argument("--remove_first_column", help="Remove first column of data")
    parser.add_argument("--epochs", help="total number of epochs")
    parser.add_argument("--batch_size", help="size of a batch")
    parser.add_argument("--learning_rate", help="learning rate if using SGD")
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = get_data(args.filepath, args.filename)
    train_ds = Dataset(train_x,train_y) # a dataset just gives a __getitem__
    test_ds = Dataset(test_x,test_y)

    train_dl = DataLoader(dataset=train_ds,batch_size=int(args.batch_size))
    # a dataloader does 2 things: shuffles the data and convert values to a tensor
    test_dl = DataLoader(dataset=test_ds,batch_size=2*int(args.batch_size))

    no_of_classes = 1 + int(test_y.max().item())
    model = LogisticClassification([len(train_x[0]), 100], [100, no_of_classes], int(args.batch_size))
    optim = Optimizer(model.trainable_params,float(args.learning_rate))
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    callbacks = [LossAndAccuracyCallback(),LrScheduler(cosine_schedule(float(args.learning_rate),0.001)),LrRecorder()]
    runner = Runner(model,optim,train_dl,test_dl,callbacks)
    runner.fit(epochs)


if __name__ == "__main__":
    main()
