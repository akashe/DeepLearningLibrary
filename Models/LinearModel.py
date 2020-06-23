from CostFunctions.MeanSquarredError import MeanSquarredError
import torch
import sys


class LinearModel(object):
    def __init__(self, input_dim, batch_size, loss_type, update_rule, learning_rate=0.01, regularization=None):
        # define the variable theta and tensors for corresponding gradients
        # by default I will update parameters using matrix form and not gradient descent
        # Implementing normal form first
        #
        # parameter_dimensions = input_dim + 1 to keep the bias term
        if loss_type == "MSE":
            self.loss = MeanSquarredError
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.update_rule = update_rule
        self.parameters = torch.randn(self.input_dim+1,1,dtype=torch.float)
        self.inputs = None
        self.targets = None
        self.labels = None

    def calculate_loss(self):
        # given the loss type calculate it
        # print loss value
        # call update_gradients
        loss_value = self.loss(self.labels,self.targets)
        print("Loss value at " + str(loss_value))
        self.update_with_gradients()

    def update_with_gradients(self):
        # update values of parameter
        if self.update_rule == "Matrix" or self.update_rule == "Closed form":
            # parameters = (X_transpose*X)_inverse * X_transpose* Y
            self.parameters = self.inputs.t().mm(self.inputs).inverse().mm(self.inputs.t()).mm(self.labels)
        if self.update_rule == "GD" or self.update_rule == "SGD":
            print("Not yet implemented")
            sys.exit(0)

    def run(self, inputs, labels):
        # calculate target values
        # call calculate loss
        # keeping bias term included in self.parameters and keeping the bias terms is inputs

        #Append inputs with 1 here
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs[0]) == self.input_dim + 1

        self.targets = inputs.mm(self.parameters)
        self.calculate_loss()

