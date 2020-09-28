from abc import ABC

from .module import Module


class LogSoftmax(Module, ABC):
    def __init__(self):
        super(LogSoftmax, self).__init__()

    def forward(self, input, *args):
        """

        Its generally better to avoid e^x because it results in inconsistency with big floating numbers
        So we have to avoid this option:
        input - input.exp().sum(-1,keepdims=True).log()
        Instead we use logSumExp trick:

        log(\sum_{i}{n}(e^{x_i}) = log(e^a \sum_{i}{n}e^{x_i - a}) = a + log(\sum_{i}{n}(e^{x_i -a})
        """
        return input - input.logsumexp(-1, keepdim=True)
