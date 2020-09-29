import torch

from .SGD import SGD

class Optimizer:
    """
    The current design is like that once entire backward pass is completed then we update the params
    with their accumulated gradients. Wouldn't work in a circular/recursive/continuous architecture.
    """

    def __init__(self,params,lr):
        self.params = params
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for i in self.params:
                i -= i.grad*self.lr

    def zero_grad(self):
        for i in self.params:
            i.grad.data.zero_()

