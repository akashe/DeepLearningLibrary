from CostFunctions.MeanSquarredError import MeanSquarredError
from CostFunctions.MeanSquarredError import MeanSquarredErrorL2
from CostFunctions.MeanSquarredError import MeanSquarredErrorL1
import torch
import sys
from TorchFunctions.dataModifications import appendOnes
from Optimizers.SGD import SGD
import math


class LinearModel(object):
    def __init__(self, input_dim, batch_size, loss_type, update_rule, learning_rate=0.001, regularization=None,
                 regularizing_constant=0.01):
        # define the variable theta and tensors for corresponding gradients
        # by default I will update parameters using matrix form and not gradient descent
        # Implementing normal form first
        #
        # parameter_dimensions = input_dim + 1 to keep the bias term
        if loss_type == "MSE" and regularization == None:
            self.loss = MeanSquarredError
        elif regularization == "L2":
            self.loss = MeanSquarredErrorL2
        elif regularization == "L1":
            self.loss = MeanSquarredErrorL1
        self.input_dim = input_dim
        self.regularization = regularization
        self.batch_size = int(batch_size)
        self.update_rule = update_rule
        self.parameters = torch.randn([self.input_dim + 1, 1], dtype=torch.float)*math.sqrt(2/(self.input_dim + 1))
        self.inputs = None
        self.targets = None
        self.labels = None
        self.learning_rate = learning_rate
        self.regularizing_constant = regularizing_constant

    def calculate_loss(self, update_parameter=True):
        # given the loss type calculate it
        # print loss value
        # call update_gradients
        loss_value = self.loss(labels=self.labels, targets=self.targets, parameters=self.parameters, regularizing_constant=self.regularizing_constant, batch_size=self.batch_size)
        print(str(loss_value))
        if update_parameter:
            self.update_value_of_parameters()

    def update_value_of_parameters(self):
        # update values of parameter
        if self.update_rule == "Matrix" or self.update_rule == "Closed form":
            if not self.regularization:
                # parameters = (X_transpose*X)_inverse * X_transpose* Y
                self.parameters = (self.inputs.t().mm(self.inputs)).inverse().mm(self.inputs.t()).mm(
                    self.labels.reshape([-1, 1]))
            if self.regularization == "L2":
                # parameters = (X_transpose*X+ delta_squared* Identity_matrix)_inverse * X_transpose* Y
                self.parameters = (self.inputs.t().mm(
                    self.inputs) + self.regularizing_constant * self.regularizing_constant * torch.eye(
                    self.input_dim + 1)).inverse().mm(self.inputs.t()).mm(self.labels.reshape([-1, 1]))
        if self.update_rule == "GD" or self.update_rule == "SGD":
            if not self.regularization:
                gradients = -(2 / self.batch_size) * self.inputs.t().mm((self.labels - self.targets).reshape([-1, 1]))
                '''
                for normal/steepest/batch gradient descent we add the value of gradients over the total length of data
                for stochastic gradient descent we can work with every single data point as an approximation of gradient
                for mini batch gradient descent we just add the values for of gradients for the number of examples in the mini batch
                Also,
                I chose my cost func as Mean Squared error so the 1/n or 1/batch_size term will end up in my gradients
                '''
            if self.regularization == "L2":
                gradients = -(2 / self.batch_size) * self.inputs.t().mm((self.labels - self.targets).reshape([-1, 1])) + \
                            self.regularizing_constant * self.regularizing_constant * 2 * self.parameters
            if self.regularization == "L1":
                gradients = -(2 / self.batch_size) * self.inputs.t().mm((self.labels - self.targets).reshape([-1, 1])) + \
                            self.regularizing_constant * torch.sign(
                    self.parameters)  # USing naive update rule
            self.parameters = SGD(self.parameters, gradients, self.learning_rate)

    def run(self, inputs, labels, update_parameter=True):
        # calculate target values
        # call calculate loss
        # keeping bias term included in self.parameters and keeping the bias terms is inputs

        self.inputs = inputs
        self.labels = labels

        # Appending inputs with 1 here
        self.inputs = appendOnes(self.inputs)

        assert len(self.inputs[0]) == self.input_dim + 1

        self.targets = self.inputs.mm(self.parameters).squeeze()
        self.calculate_loss(update_parameter)
