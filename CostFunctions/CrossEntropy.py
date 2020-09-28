from abc import ABC
from .CostFunction import CostFunction
from Modules import LogSoftmax


class CrossEntropy(CostFunction, ABC):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.lsf = LogSoftmax()

    def forward(self, inputs, targets, *args):
        inputs = self.lsf(inputs)
        batch_size = len(targets)
        # TODO: is there a better way to avoid input.o, one way cud be to make a NLL module not NLL costfunction
        if len(targets.shape) == 1:
            # Target is 1D tensor
            return - inputs.o[range(batch_size), targets.long()].mean()[None]
        else:
            assert inputs.o.shape == targets.shape
            return - inputs.o[range(batch_size), targets.argmax(dim=1)].mean()[None]
