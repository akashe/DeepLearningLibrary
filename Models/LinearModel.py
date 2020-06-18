from CostFunctions.MeanSquarredError import MeanSquarredError


class LinearModel(object):
    def __init__(self, loss_type, rate_of_learning):
        # define the variable theta and tensors for corresponding gradients
        pass

    def calculate_loss(self, labels, targets):
        # given the loss type calculate it
        # print loss value
        # call update_gradients
        pass

    def update_with_gradients(self):
        # update values of theta
        pass

    def run(self, inputs, labels):
        # calculate target values
        # call calculate loss
        pass
