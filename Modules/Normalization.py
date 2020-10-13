from abc import ABC
import torch
from .module import Module
from TorchFunctions.dataInitialization import kaiming_initialization


class BatchNorm(Module, ABC):
    def __init__(self, momentum=0.9, epsilon=1e-5):
        '''
        BatchNorm normalizes the values across a batch, so lets say a particular x value is
        higher than normal then this becomes a prominent feature after normalization

        :param input_shape: shape of the form [batch,....]
        :param momentum: momentum value for running averaging
        :param epsilon: epsilon value to avoid zero variance while normalizing
        '''
        super(BatchNorm, self).__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.first_call = 0  # Necessary coz we don't have input shape before running
        # Have to initialize here or else optimizer wont get these for updates
        self.gamma = torch.ones(1, requires_grad=True)
        self.beta = torch.zeros(1, requires_grad=True)

    def initialize_more_params(self, input_shape):
        with torch.no_grad():
            self.gamma.data = torch.ones(size=(1, *input_shape[1:]))
            # kaiming_initialization(self.gamma)
            # Other option to input_shape could have been just the no of features but I want to make it general
            self.beta.data = torch.zeros(size=(1, *input_shape[1:]))

        # exponential averages
        self.mean = torch.zeros(size=(1, *input_shape[1:]))
        self.var = torch.ones(size=(1, *input_shape[1:]))

    def update_means_and_vars(self, input_):
        mean_ = input_.mean(0, keepdim=True)
        variance_ = input_.var(0, keepdim=True)
        '''
        Running average: 
        self.mean = self.momentum*self.mean + (1-self.momentum)*mean_
        using lerp_ coz it might be faster
        '''
        self.mean.lerp_(mean_, (1 - self.momentum))  # its the way lerp_ is defined
        self.var.lerp_(variance_, (1 - self.momentum))
        return mean_, variance_

    def forward(self, input, *args):
        if self.first_call == 0:
            self.initialize_more_params(input.size())
            self.first_call += 1

        if self.train_:
            m, v = self.update_means_and_vars(input)
        else:
            m, v = self.mean, self.var

        input = (input - m) / (v + self.epsilon).sqrt()

        return self.gamma * input + self.beta


class InstanceNorm(Module, ABC):
    pass


class LayerNorm(Module, ABC):
    pass
