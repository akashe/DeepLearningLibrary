import torch

from .SGD import SGD


class SGDOptimizerForModules:
    '''
    This class will support the module and model structure in this library
    '''

    def __init__(self, lr):
        self.lr = lr
        # Not keeping params here as the same object will be used by different modules with different params

    def step(self, params, grads):
        with torch.no_grad():
            for i, j in zip(params, grads):
                i -= self.lr * j

    def zero_grad(self, params):
        with torch.no_grad():
            for i in params:
                # i.grad = torch.zeros(i.size())
                i.grad.data.zero_()

class Optimizer:
    """
    The current design is like that once entire backward pass is completed then we update the params
    with their accumulated gradients. Wouldn't work in a circular/recursive/continuous architecture.

    This class is for pytorch type design.
    """

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for i in self.params:
                i -= i.grad * self.lr

    def zero_grad(self):
        for i in self.params:
            i.grad.data.zero_()
