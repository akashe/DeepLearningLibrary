from abc import ABC
from .CostFunction import CostFunction


class NLL(CostFunction, ABC):
    def __init__(self):
        super(NLL, self).__init__()

    def forward(self, input, target, *args):
        '''
        Requires one hot encoding in the target

        :param input: log softmax values from previous layer
        :param target: 2D tensor with one hot encoding of target values
        :param args:
        :return: neg log likelihood error for the given input and targets
        '''

        assert input.shape == target.shape
        batch_size = len(target)
        return - input[range(batch_size), target.argmax(dim=1)].mean()[None]
